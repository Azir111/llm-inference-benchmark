"""
负载均衡策略对比压测：round_robin vs least_conn
新建独立文件，不修改原有 load_test.py

用法：
  python benchmark/load_test_lb.py
  python benchmark/load_test_lb.py --concurrency 1 4 8 16 --requests 50
"""

import asyncio
import aiohttp
import time
import argparse
import json
import random
import statistics
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import List
from pathlib import Path

# ── 复用原有问题集 ──────────────────────────────────────
from load_test import ECOMMERCE_QUESTIONS, RequestResult

# ── 两个NGINX端口（和 deploy/nginx.conf 对应） ───────────
ROUND_ROBIN_URL = "http://localhost:9090/v1/chat/completions"
LEAST_CONN_URL  = "http://localhost:9091/v1/chat/completions"


@dataclass
class LBBenchmarkResult:
    """负载均衡压测结果，和原有BenchmarkResult结构对齐"""
    strategy: str          # round_robin / least_conn
    concurrency: int
    total_requests: int
    success_count: int
    error_count: int
    latency_mean: float = 0.0
    latency_p50: float  = 0.0
    latency_p90: float  = 0.0
    latency_p99: float  = 0.0
    latency_min: float  = 0.0
    latency_max: float  = 0.0
    qps: float = 0.0
    total_tokens_per_sec: float = 0.0
    raw_results: List[RequestResult] = field(default_factory=list, repr=False)

    def compute_stats(self):
        """和原有 BenchmarkResult.compute_stats() 保持一致"""
        latencies = [r.latency_ms for r in self.raw_results if r.success]
        if not latencies:
            return
        s = sorted(latencies)
        n = len(s)
        self.latency_mean = round(statistics.mean(latencies), 1)
        self.latency_p50  = round(s[int(n * 0.50)], 1)
        self.latency_p90  = round(s[int(n * 0.90)], 1)
        self.latency_p99  = round(s[min(int(n * 0.99), n - 1)], 1)
        self.latency_min  = round(s[0], 1)
        self.latency_max  = round(s[-1], 1)
        tps_list = [r.tokens_per_sec for r in self.raw_results
                    if r.success and r.tokens_per_sec > 0]
        if tps_list:
            self.total_tokens_per_sec = round(sum(tps_list), 1)

    def to_dict(self):
        d = asdict(self)
        d.pop("raw_results", None)
        return d


# ── 单次请求（复用原有逻辑，只改函数签名） ────────────────
async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    question: str,
    max_tokens: int,
    timeout: int,
    concurrency: int,
) -> RequestResult:
    payload = {
        "model": "customer-service-llm",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status != 200:
                return RequestResult(False, latency_ms,
                                     error=f"HTTP {resp.status}",
                                     concurrency=concurrency)
            data = await resp.json()
            usage = data.get("usage", {})
            n_tokens = usage.get("completion_tokens", 0)
            tps = n_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            return RequestResult(True, latency_ms, n_tokens, tps,
                                 concurrency=concurrency)
    except asyncio.TimeoutError:
        return RequestResult(False, timeout * 1000,
                             error="Timeout", concurrency=concurrency)
    except Exception as e:
        return RequestResult(False, (time.perf_counter() - t0) * 1000,
                             error=str(e), concurrency=concurrency)


# ── 单轮压测（和原有 run_concurrent_test 结构对齐） ────────
async def run_lb_test(
    url: str,
    strategy: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int = 200,
    timeout: int = 60,
) -> LBBenchmarkResult:
    print(f"  [{strategy:12s}] 并发={concurrency}, 总请求={num_requests}")

    result = LBBenchmarkResult(
        strategy=strategy,
        concurrency=concurrency,
        total_requests=num_requests,
        success_count=0,
        error_count=0,
    )

    semaphore = asyncio.Semaphore(concurrency)
    questions = [random.choice(ECOMMERCE_QUESTIONS) for _ in range(num_requests)]

    async def bounded_request(q: str) -> RequestResult:
        async with semaphore:
            return await single_request(
                session, url, q, max_tokens, timeout, concurrency
            )

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        wall_start = time.perf_counter()
        raw = await asyncio.gather(*[bounded_request(q) for q in questions])
        wall_time = time.perf_counter() - wall_start

    result.raw_results   = list(raw)
    result.success_count = sum(1 for r in raw if r.success)
    result.error_count   = sum(1 for r in raw if not r.success)
    result.qps           = round(result.success_count / wall_time, 2)
    result.compute_stats()

    print(f"    成功: {result.success_count}/{num_requests}  "
          f"QPS: {result.qps}  "
          f"P50: {result.latency_p50}ms  "
          f"P90: {result.latency_p90}ms  "
          f"Token/s: {result.total_tokens_per_sec}")

    return result


# ── 健康检查（和原有 check_server 保持一致） ──────────────
def check_server(url: str, name: str) -> bool:
    health_url = url.replace("/v1/chat/completions", "/v1/models")
    try:
        urllib.request.urlopen(health_url, timeout=5)
        print(f"  [OK] {name} 就绪: {url}")
        return True
    except Exception as e:
        print(f"  [WARN] {name} 不可达: {health_url} ({e})")
        return False


# ── 完整对比流程 ───────────────────────────────────────
async def run_lb_benchmark(
    concurrency_levels: List[int],
    requests_per_level: int,
    max_tokens: int,
    timeout: int,
    warmup_requests: int,
) -> List[LBBenchmarkResult]:

    all_results = []

    for strategy, url in [("round_robin", ROUND_ROBIN_URL),
                           ("least_conn",  LEAST_CONN_URL)]:
        print(f"\n{'='*60}")
        print(f"策略: {strategy.upper()}")
        print(f"{'='*60}")

        # 预热（和原有保持一致）
        if warmup_requests > 0:
            print(f"\n[预热] 发送 {warmup_requests} 条请求...")
            await run_lb_test(url, strategy, concurrency=1,
                              num_requests=warmup_requests,
                              max_tokens=max_tokens, timeout=timeout)
            print("  预热完成，等待 2s...")
            await asyncio.sleep(2)

        print(f"\n[正式压测]")
        for c in concurrency_levels:
            r = await run_lb_test(url, strategy, concurrency=c,
                                  num_requests=requests_per_level,
                                  max_tokens=max_tokens, timeout=timeout)
            all_results.append(r)
            await asyncio.sleep(1)

    # 打印对比摘要
    print(f"\n{'='*60}")
    print("对比摘要：least_conn 相对 round_robin 的提升")
    print(f"{'='*60}")
    rr_map = {r.concurrency: r for r in all_results if r.strategy == "round_robin"}
    lc_map = {r.concurrency: r for r in all_results if r.strategy == "least_conn"}
    print(f"{'并发':>4}  {'QPS提升':>8}  {'P90降低':>8}  {'成功率RR':>8}  {'成功率LC':>8}")
    for c in concurrency_levels:
        rr, lc = rr_map[c], lc_map[c]
        qps_gain = (lc.qps - rr.qps) / rr.qps * 100 if rr.qps else 0
        p90_drop = (rr.latency_p90 - lc.latency_p90) / rr.latency_p90 * 100 \
                   if rr.latency_p90 else 0
        rr_sr = f"{rr.success_count/rr.total_requests*100:.1f}%"
        lc_sr = f"{lc.success_count/lc.total_requests*100:.1f}%"
        print(f"{c:>4}  {qps_gain:>+7.1f}%  {p90_drop:>7.1f}%  {rr_sr:>8}  {lc_sr:>8}")

    return all_results


# ── 入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="负载均衡策略对比压测")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--requests",    type=int, default=50)
    parser.add_argument("--max-tokens",  type=int, default=200)
    parser.add_argument("--timeout",     type=int, default=60)
    parser.add_argument("--warmup",      type=int, default=5)
    parser.add_argument("--output", default="results/lb_comparison.json")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("负载均衡策略对比压测：round_robin vs least_conn")
    print("="*60)
    print(f"  并发级别: {args.concurrency}")
    print(f"  每级请求: {args.requests}")

    # 健康检查
    print("\n[检查服务]")
    if not check_server(ROUND_ROBIN_URL, "NGINX round_robin"):
        print("  请先启动 NGINX，运行 sudo nginx -c deploy/nginx.conf")
        exit(1)
    if not check_server(LEAST_CONN_URL, "NGINX least_conn"):
        print("  请检查 NGINX 配置，9091 端口未就绪")
        exit(1)

    results = asyncio.run(run_lb_benchmark(
        concurrency_levels=args.concurrency,
        requests_per_level=args.requests,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        warmup_requests=args.warmup,
    ))

    # 保存结果（和原有 raw_results.json 格式一致）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 结果已保存到 {args.output}")
    print("     运行 python benchmark/analyze.py 生成图表和报告")