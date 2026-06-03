#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式并发压测脚本（OpenAI-compatible SSE）

相比端到端延迟，LLM serving 真正关心的是“逐 token”维度的指标：
    - TTFT (Time To First Token)：首 token 延迟，决定“响应快不快”，受 prefill + 排队影响
    - TPOT (Time Per Output Token)：每个输出 token 的平均间隔，决定“吐字流不流畅”
    - ITL  (Inter-Token Latency)：相邻 token 到达间隔的分布（TPOT 是它的均值）

只测端到端延迟会被输出长度污染（500 token 的回答 vs 50 token 的回答没法比），
而 vLLM 的 Continuous Batching 优势恰恰体现在高并发下 TTFT 不爆炸 —— 必须流式才测得到。

用法示例：
    python load_test_stream.py \\
        --vllm-url     http://localhost:8000/v1/chat/completions \\
        --baseline-url http://localhost:8001/v1/chat/completions \\
        --model        customer-service-llm \\
        --concurrency  1 2 4 8 16 \\
        --requests     50 \\
        --warmup       5 \\
        --max-tokens   256

只压一个后端时省略 --baseline-url 即可。
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import aiohttp


# ----------------------------------------------------------------------------
# 默认压测 prompt（电商客服场景）。也可用 --prompt-file 传入一个 txt，一行一个 prompt。
# ----------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "我买的连衣裙收到后发现尺码偏小，想换大一码，怎么操作？",
    "订单显示已发货三天了物流一直没更新，是不是丢件了？",
    "我用了优惠券下单，现在想退其中一件，优惠会怎么算？",
    "你们的会员积分多久过期？过期的还能恢复吗？",
    "下单后能不能修改收货地址？我填错小区了。",
    "这个充电宝能带上飞机吗？容量是多少毫安？",
    "申请了退款为什么还要我先把货寄回去？运费谁出？",
    "我想批量采购 50 件做企业团购，有折扣吗？开发票吗？",
    "收到的商品有破损，已经拍照了，理赔流程是怎样的？",
    "你们家的羽绒服洗了会不会跑绒？怎么保养？",
]


# ----------------------------------------------------------------------------
# 单次请求结果
# ----------------------------------------------------------------------------
@dataclass
class RequestResult:
    success: bool
    ttft_ms: Optional[float] = None          # 首 token 延迟
    e2e_ms: Optional[float] = None           # 端到端：请求发出 -> 最后一个 token
    tpot_ms: Optional[float] = None          # (e2e - ttft) / (output_tokens - 1)
    output_tokens: int = 0                   # 输出 token 数（优先用 usage，否则数 chunk）
    itls_ms: list = field(default_factory=list)  # 相邻 token 间隔序列
    error: Optional[str] = None


# ----------------------------------------------------------------------------
# 单个流式请求：边收边记时间戳
# ----------------------------------------------------------------------------
async def stream_one_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        # vLLM / 多数 OpenAI 兼容后端支持，最后会多推一个带 usage 的 chunk，token 数最准
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: Optional[float] = None
    t_prev: Optional[float] = None
    itls: list[float] = []
    chunk_token_count = 0          # 用“有内容的 chunk 数”兜底估算 token 数
    usage_tokens: Optional[int] = None

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                body = (await resp.text())[:200]
                return RequestResult(False, error=f"HTTP {resp.status}: {body}")

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break

                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                # 最后的 usage chunk：choices 通常为空，但带准确 token 数
                usage = obj.get("usage")
                if usage and usage.get("completion_tokens") is not None:
                    usage_tokens = usage["completion_tokens"]

                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if not content:
                    continue

                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                else:
                    itls.append((now - t_prev) * 1000.0)
                t_prev = now
                chunk_token_count += 1

        t_end = time.perf_counter()

        if t_first is None:
            return RequestResult(False, error="无内容返回（空响应）")

        out_tokens = usage_tokens if usage_tokens is not None else chunk_token_count
        ttft_ms = (t_first - t_start) * 1000.0
        e2e_ms = (t_end - t_start) * 1000.0
        # TPOT 按惯例排除首 token：剩余生成时间 / (输出 token 数 - 1)
        tpot_ms = (e2e_ms - ttft_ms) / (out_tokens - 1) if out_tokens > 1 else None

        return RequestResult(
            success=True,
            ttft_ms=ttft_ms,
            e2e_ms=e2e_ms,
            tpot_ms=tpot_ms,
            output_tokens=out_tokens,
            itls_ms=itls,
        )

    except asyncio.TimeoutError:
        return RequestResult(False, error=f"超时 (>{timeout_s}s)")
    except aiohttp.ClientError as e:
        return RequestResult(False, error=f"连接错误: {e}")
    except Exception as e:  # noqa: BLE001  压测脚本兜底，单请求异常不影响整体
        return RequestResult(False, error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------------------
# 跑一个并发级别
# ----------------------------------------------------------------------------
async def run_level(
    url: str,
    model: str,
    prompts: list[str],
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(idx: int, session) -> RequestResult:
        async with semaphore:  # 精确控制在途请求数 == concurrency
            prompt = prompts[idx % len(prompts)]
            return await stream_one_request(session, url, model, prompt, max_tokens, timeout_s)

    # 整个并发级别共用一个 session（连接池），wall_clock 用于算 QPS / 系统吞吐
    connector = aiohttp.TCPConnector(limit=0)  # 不额外限制，交给 Semaphore 控
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(worker(i, session)) for i in range(n_requests)]
        results = await asyncio.gather(*tasks)
        wall = time.perf_counter() - t0

    return aggregate(results, concurrency, wall)


# ----------------------------------------------------------------------------
# 聚合统计
# ----------------------------------------------------------------------------
def percentile(values: list[float], p: float) -> float:
    """线性插值百分位，避免引入 numpy 依赖。"""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def aggregate(results: list[RequestResult], concurrency: int, wall_s: float) -> dict:
    ok = [r for r in results if r.success]
    n_total = len(results)
    n_ok = len(ok)

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in ok]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    all_itls = [x for r in ok for x in r.itls_ms]
    total_out_tokens = sum(r.output_tokens for r in ok)

    # 失败原因统计，方便定位（区分超时 / HTTP 错误等）
    err_counts: dict[str, int] = {}
    for r in results:
        if not r.success and r.error:
            key = r.error.split(":")[0]
            err_counts[key] = err_counts.get(key, 0) + 1

    return {
        "concurrency": concurrency,
        "requests": n_total,
        "success": n_ok,
        "success_rate": round(n_ok / n_total * 100, 1) if n_total else 0.0,
        "wall_s": round(wall_s, 2),
        # 系统级吞吐
        "qps": round(n_ok / wall_s, 2) if wall_s > 0 else 0.0,
        "output_token_throughput": round(total_out_tokens / wall_s, 1) if wall_s > 0 else 0.0,
        # TTFT
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p90_ms": round(percentile(ttfts, 90), 1),
        "ttft_p99_ms": round(percentile(ttfts, 99), 1),
        # TPOT（每输出 token 间隔）
        "tpot_mean_ms": round(sum(tpots) / len(tpots), 1) if tpots else 0.0,
        "tpot_p90_ms": round(percentile(tpots, 90), 1),
        # ITL 长尾（卡顿）
        "itl_p99_ms": round(percentile(all_itls, 99), 1),
        # 端到端
        "e2e_p50_ms": round(percentile(e2es, 50), 1),
        "e2e_p90_ms": round(percentile(e2es, 90), 1),
        "avg_output_tokens": round(total_out_tokens / n_ok, 1) if n_ok else 0.0,
        "errors": err_counts,
    }


# ----------------------------------------------------------------------------
# 单个后端：预热 + 各并发级别
# ----------------------------------------------------------------------------
async def benchmark_backend(
    name: str,
    url: str,
    model: str,
    prompts: list[str],
    levels: list[int],
    n_requests: int,
    warmup: int,
    max_tokens: int,
    timeout_s: float,
) -> list[dict]:
    print(f"\n{'='*70}\n后端: {name}  ({url})\n{'='*70}")

    if warmup > 0:
        print(f"预热 {warmup} 次（结果丢弃，排除冷启动 / 编译缓存影响）...")
        async with aiohttp.ClientSession() as s:
            for i in range(warmup):
                await stream_one_request(s, url, model, prompts[i % len(prompts)],
                                         max_tokens, timeout_s)

    rows = []
    for c in levels:
        print(f"  并发={c:<3} 请求={n_requests} ... ", end="", flush=True)
        row = await run_level(url, model, prompts, c, n_requests, max_tokens, timeout_s)
        rows.append(row)
        print(
            f"QPS={row['qps']:<6} "
            f"TTFT_P90={row['ttft_p90_ms']:<7}ms  "
            f"TPOT_mean={row['tpot_mean_ms']:<6}ms  "
            f"成功率={row['success_rate']}%"
        )
    return rows


def print_table(name: str, rows: list[dict]) -> None:
    print(f"\n--- {name} ---")
    header = (f"{'并发':>4} | {'QPS':>6} | {'tok/s':>7} | {'TTFT_P50':>8} | "
              f"{'TTFT_P90':>8} | {'TPOT_mean':>9} | {'ITL_P99':>7} | {'成功率':>6}")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['concurrency']:>4} | {r['qps']:>6} | {r['output_token_throughput']:>7} | "
              f"{r['ttft_p50_ms']:>8} | {r['ttft_p90_ms']:>8} | {r['tpot_mean_ms']:>9} | "
              f"{r['itl_p99_ms']:>7} | {r['success_rate']:>5}%")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="流式并发压测（TTFT / TPOT / ITL）")
    p.add_argument("--vllm-url", required=True, help="vLLM 的 chat/completions URL")
    p.add_argument("--baseline-url", default=None, help="对照组 URL（可选）")
    p.add_argument("--model", default="customer-service-llm", help="served-model-name")
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--requests", type=int, default=50, help="每个并发级别的总请求数")
    p.add_argument("--warmup", type=int, default=5, help="预热请求数（丢弃）")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--timeout", type=float, default=120.0, help="单请求超时（秒）")
    p.add_argument("--prompt-file", default=None, help="自定义 prompt 文件，一行一个")
    p.add_argument("--output", default="results/stream_results.json")
    return p.parse_args()


async def main():
    args = parse_args()

    if args.prompt_file:
        prompts = [l.strip() for l in Path(args.prompt_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    out = {
        "config": {
            "model": args.model,
            "concurrency": args.concurrency,
            "requests_per_level": args.requests,
            "warmup": args.warmup,
            "max_tokens": args.max_tokens,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "backends": {},
    }

    vllm_rows = await benchmark_backend(
        "vLLM", args.vllm_url, args.model, prompts, args.concurrency,
        args.requests, args.warmup, args.max_tokens, args.timeout,
    )
    out["backends"]["vllm"] = vllm_rows

    if args.baseline_url:
        base_rows = await benchmark_backend(
            "Transformers (baseline)", args.baseline_url, args.model, prompts,
            args.concurrency, args.requests, args.warmup, args.max_tokens, args.timeout,
        )
        out["backends"]["baseline"] = base_rows

    # 汇总表
    print("\n" + "=" * 70 + "\n汇总\n" + "=" * 70)
    print_table("vLLM", vllm_rows)
    if args.baseline_url:
        print_table("Transformers", out["backends"]["baseline"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n原始结果已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())