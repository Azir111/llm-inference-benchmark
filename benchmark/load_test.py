"""
并发压测脚本
测试不同并发数下的吞吐量和延迟分布
支持同时压测 vLLM 和 Transformers baseline，输出对比结果
"""

import asyncio
import aiohttp
import time
import argparse
import json
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path


# ── 电商客服测试问题集（覆盖5类意图） ──────────────────────────────────
ECOMMERCE_QUESTIONS = [
    # 物流查询
    "我的订单号是2024112901，请问什么时候能到？",
    "快递显示已揽件但三天没更新，是什么情况？",
    "我在上海，下单广州发货，一般几天到？",
    "订单显示已发货但没有物流信息，怎么查？",
    "我的包裹显示派送中，但快递员没来，怎么办？",
    # 退换货
    "买的手机屏幕有坏点，怎么退货？",
    "衣服洗了一次就掉色，可以换货吗？",
    "收到的商品和图片颜色不一样，能退款吗？",
    "退货申请提交了三天了，还没处理，催一下",
    "7天无理由退货政策是怎么规定的？",
    # 商品咨询
    "这款耳机支持降噪吗？",
    "这个电饭煲是不是国家3C认证？",
    "笔记本支持多少赫兹屏幕？",
    "这件羽绒服的充绒量是多少？",
    "手机壳适合iPhone15 Pro Max吗？",
    # 投诉建议
    "客服态度太差了，我要投诉！",
    "商品描述严重不符，这是虚假宣传！",
    "发货太慢了，下单5天才发货，太不专业",
    "建议增加7×24小时人工客服",
    "希望能支持货到付款",
    # 售后问题
    "保修期内耳机坏了，怎么维修？",
    "购买的电器出现故障，如何申请免费上门维修？",
    "延长保修服务怎么购买？",
    "收到破损商品怎么索赔？",
    "发票怎么申请？",
]


@dataclass
class RequestResult:
    """单次请求的结果"""
    success: bool
    latency_ms: float
    tokens_generated: int = 0
    tokens_per_sec: float = 0.0
    error: str = ""
    concurrency: int = 0


@dataclass
class BenchmarkResult:
    """一组并发测试的汇总结果"""
    backend: str
    concurrency: int
    total_requests: int
    success_count: int
    error_count: int
    # 延迟统计 (ms)
    latency_mean: float = 0.0
    latency_p50: float = 0.0
    latency_p90: float = 0.0
    latency_p99: float = 0.0
    latency_min: float = 0.0
    latency_max: float = 0.0
    # 吞吐量
    qps: float = 0.0                  # Queries Per Second
    total_tokens_per_sec: float = 0.0  # 总 token 生成速度
    # 原始请求列表（不序列化到 JSON）
    raw_results: List[RequestResult] = field(default_factory=list, repr=False)

    def compute_stats(self):
        """从原始结果计算统计量"""
        latencies = [r.latency_ms for r in self.raw_results if r.success]
        if not latencies:
            return
        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        self.latency_mean = round(statistics.mean(latencies), 1)
        self.latency_p50 = round(latencies_sorted[int(n * 0.50)], 1)
        self.latency_p90 = round(latencies_sorted[int(n * 0.90)], 1)
        self.latency_p99 = round(latencies_sorted[min(int(n * 0.99), n - 1)], 1)
        self.latency_min = round(latencies_sorted[0], 1)
        self.latency_max = round(latencies_sorted[-1], 1)
        tps_list = [r.tokens_per_sec for r in self.raw_results if r.success and r.tokens_per_sec > 0]
        if tps_list:
            self.total_tokens_per_sec = round(sum(tps_list), 1)

    def to_dict(self):
        d = asdict(self)
        d.pop("raw_results", None)
        return d


async def single_request(
    session: aiohttp.ClientSession,
    url: str,
    question: str,
    max_tokens: int,
    timeout: int,
    concurrency: int,
) -> RequestResult:
    """发送单个请求并返回结果"""
    payload = {
        "model": "customer-service-llm",
        "messages": [{"role": "user", "content": question}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    t0 = time.perf_counter()
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            latency_ms = (time.perf_counter() - t0) * 1000
            if resp.status != 200:
                return RequestResult(
                    success=False,
                    latency_ms=latency_ms,
                    error=f"HTTP {resp.status}",
                    concurrency=concurrency,
                )
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            n_tokens = usage.get("completion_tokens", 0)
            tokens_per_sec = n_tokens / (latency_ms / 1000) if latency_ms > 0 else 0
            return RequestResult(
                success=True,
                latency_ms=latency_ms,
                tokens_generated=n_tokens,
                tokens_per_sec=tokens_per_sec,
                concurrency=concurrency,
            )
    except asyncio.TimeoutError:
        return RequestResult(
            success=False,
            latency_ms=timeout * 1000,
            error="Timeout",
            concurrency=concurrency,
        )
    except Exception as e:
        return RequestResult(
            success=False,
            latency_ms=(time.perf_counter() - t0) * 1000,
            error=str(e),
            concurrency=concurrency,
        )


async def run_concurrent_test(
    url: str,
    backend_name: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int = 200,
    timeout: int = 60,
) -> BenchmarkResult:
    """
    运行一轮并发测试

    Args:
        url: 推理服务地址，如 http://localhost:8000/v1/chat/completions
        backend_name: 后端名称（vllm / transformers）
        concurrency: 并发数
        num_requests: 总请求数
        max_tokens: 最大生成 token 数
        timeout: 单请求超时（秒）
    """
    print(f"\n  [{backend_name}] 并发={concurrency}, 总请求={num_requests}")

    result = BenchmarkResult(
        backend=backend_name,
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
        tasks = [bounded_request(q) for q in questions]
        raw = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    result.raw_results = list(raw)
    result.success_count = sum(1 for r in raw if r.success)
    result.error_count = sum(1 for r in raw if not r.success)
    result.qps = round(result.success_count / wall_time, 2)
    result.compute_stats()

    # 打印简要结果
    print(f"    成功: {result.success_count}/{num_requests}  "
          f"QPS: {result.qps}  "
          f"P50: {result.latency_p50}ms  "
          f"P90: {result.latency_p90}ms  "
          f"Token/s: {result.total_tokens_per_sec}")

    return result


async def run_full_benchmark(
    vllm_url: Optional[str],
    baseline_url: Optional[str],
    concurrency_levels: List[int],
    requests_per_level: int,
    max_tokens: int,
    timeout: int,
    warmup_requests: int,
) -> List[BenchmarkResult]:
    """完整压测流程：预热 → 多并发级别测试"""
    all_results = []

    backends = []
    if vllm_url:
        backends.append(("vllm", vllm_url))
    if baseline_url:
        backends.append(("transformers", baseline_url))

    for backend_name, url in backends:
        print(f"\n{'='*60}")
        print(f"压测后端: {backend_name.upper()}")
        print(f"{'='*60}")

        # 预热
        if warmup_requests > 0:
            print(f"\n[预热] 发送 {warmup_requests} 条请求...")
            await run_concurrent_test(
                url, backend_name, concurrency=1,
                num_requests=warmup_requests, max_tokens=max_tokens, timeout=timeout
            )
            print("  预热完成，等待 2s...")
            await asyncio.sleep(2)

        # 正式测试
        print(f"\n[正式压测]")
        for c in concurrency_levels:
            r = await run_concurrent_test(
                url, backend_name, concurrency=c,
                num_requests=requests_per_level,
                max_tokens=max_tokens, timeout=timeout
            )
            all_results.append(r)
            await asyncio.sleep(1)  # 两轮之间稍作冷却

    return all_results


def check_server(url: str, name: str) -> bool:
    """检查服务是否可达"""
    import urllib.request
    health_url = url.replace("/v1/chat/completions", "/health")
    try:
        urllib.request.urlopen(health_url, timeout=5)
        print(f"  [OK] {name} 服务就绪: {url}")
        return True
    except Exception as e:
        print(f"  [WARN] {name} 服务不可达: {health_url} ({e})")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 并发推理压测")
    parser.add_argument("--vllm-url", type=str,
                        default="http://localhost:8000/v1/chat/completions",
                        help="vLLM 服务地址")
    parser.add_argument("--baseline-url", type=str,
                        default="http://localhost:8001/v1/chat/completions",
                        help="Transformers baseline 服务地址")
    parser.add_argument("--skip-vllm", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--concurrency", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16],
                        help="并发级别列表，如 1 4 8 16")
    parser.add_argument("--requests", type=int, default=50,
                        help="每个并发级别的总请求数")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=5,
                        help="预热请求数（0 表示跳过）")
    parser.add_argument("--output", type=str, default="results/raw_results.json",
                        help="原始结果保存路径")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("LLM 并发推理压测")
    print("="*60)
    print(f"  并发级别: {args.concurrency}")
    print(f"  每级请求: {args.requests}")
    print(f"  最大 tokens: {args.max_tokens}")

    # 检查服务连通性
    print("\n[检查服务]")
    vllm_url = None if args.skip_vllm else args.vllm_url
    baseline_url = None if args.skip_baseline else args.baseline_url

    if vllm_url and not check_server(vllm_url, "vLLM"):
        print("  vLLM 服务未运行，跳过（可用 --skip-vllm 明确跳过）")
        vllm_url = None
    if baseline_url and not check_server(baseline_url, "Transformers"):
        print("  Transformers 服务未运行，跳过（可用 --skip-baseline 明确跳过）")
        baseline_url = None

    if not vllm_url and not baseline_url:
        print("\n[ERROR] 没有可用的服务，退出")
        exit(1)

    results = asyncio.run(
        run_full_benchmark(
            vllm_url=vllm_url,
            baseline_url=baseline_url,
            concurrency_levels=args.concurrency,
            requests_per_level=args.requests,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            warmup_requests=args.warmup,
        )
    )

    # 保存原始结果
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 原始结果已保存到 {args.output}")
    print("     运行 python benchmark/analyze.py 生成分析报告")