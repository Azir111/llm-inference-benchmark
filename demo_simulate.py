"""
快速演示脚本 —— 模拟压测数据并生成报告
适合在没有 GPU / 没有部署服务时验证分析流程
生成的数据参考 RTX 4060Ti 真实实验数值

运行：python demo_simulate.py
"""

import json
import random
import math
from pathlib import Path


def simulate_latency(base_ms: float, concurrency: int, jitter: float = 0.15) -> float:
    """模拟延迟随并发增加的变化（叠加排队延迟）"""
    queuing = base_ms * math.log1p(concurrency - 1) * 0.4
    noise = random.gauss(0, base_ms * jitter)
    return max(base_ms * 0.8, base_ms + queuing + noise)


def generate_mock_results():
    """
    生成模拟压测结果
    vLLM 参考值：单请求 ~750ms，Continuous Batching 使高并发吞吐大幅提升
    Transformers 参考值：单请求 ~1051ms，高并发几乎无吞吐提升
    """
    concurrency_levels = [1, 2, 4, 8, 16]
    results = []

    for concurrency in concurrency_levels:
        # ── vLLM ──────────────────────────────────────────────
        # Continuous Batching：QPS 随并发近线性增加（GPU 利用率高）
        vllm_base_ms = 750
        latencies = sorted([simulate_latency(vllm_base_ms, concurrency) for _ in range(50)])
        n = len(latencies)
        # vLLM 高并发下 QPS 接近线性扩展
        vllm_qps = concurrency * (1 / (vllm_base_ms / 1000)) * random.uniform(0.75, 0.92)
        results.append({
            "backend": "vllm",
            "concurrency": concurrency,
            "total_requests": 50,
            "success_count": 50,
            "error_count": 0,
            "latency_mean": round(sum(latencies) / n, 1),
            "latency_p50": round(latencies[int(n * 0.50)], 1),
            "latency_p90": round(latencies[int(n * 0.90)], 1),
            "latency_p99": round(latencies[min(int(n * 0.99), n - 1)], 1),
            "latency_min": round(latencies[0], 1),
            "latency_max": round(latencies[-1], 1),
            "qps": round(vllm_qps, 2),
            "total_tokens_per_sec": round(vllm_qps * random.uniform(55, 70), 1),
        })

        # ── Transformers ───────────────────────────────────────
        # 串行 + static：高并发下请求排队，QPS 几乎不增加
        tf_base_ms = 1051
        # 串行处理：并发越高，排队越严重，QPS 增长极慢
        tf_effective_concurrency = math.log1p(concurrency)
        latencies_tf = sorted([
            simulate_latency(tf_base_ms, concurrency, jitter=0.25) for _ in range(50)
        ])
        tf_qps = tf_effective_concurrency * (1 / (tf_base_ms / 1000)) * random.uniform(0.55, 0.70)
        results.append({
            "backend": "transformers",
            "concurrency": concurrency,
            "total_requests": 50,
            "success_count": 49 if concurrency >= 16 else 50,  # 高并发偶有超时
            "error_count": 1 if concurrency >= 16 else 0,
            "latency_mean": round(sum(latencies_tf) / len(latencies_tf), 1),
            "latency_p50": round(latencies_tf[int(n * 0.50)], 1),
            "latency_p90": round(latencies_tf[int(n * 0.90)], 1),
            "latency_p99": round(latencies_tf[min(int(n * 0.99), n - 1)], 1),
            "latency_min": round(latencies_tf[0], 1),
            "latency_max": round(latencies_tf[-1], 1),
            "qps": round(tf_qps, 2),
            "total_tokens_per_sec": round(tf_qps * random.uniform(30, 40), 1),
        })

    return results


if __name__ == "__main__":
    random.seed(42)
    print("生成模拟压测数据（基于 RTX 4060Ti 参考值）...")

    results = generate_mock_results()

    Path("results").mkdir(exist_ok=True)
    out_path = "results/raw_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[OK] 模拟数据已保存: {out_path}")
    print("\n运行分析:")
    print("  python benchmark/analyze.py")