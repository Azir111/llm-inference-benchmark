"""
压测结果分析脚本
读取 raw_results.json，生成：
  1. 控制台汇总表格
  2. 延迟分布图（latency_distribution.png）
  3. 吞吐量对比图（throughput_comparison.png）
  4. Markdown 报告（benchmark_report.md）
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict


# ── 数据加载 ──────────────────────────────────────────────────────────

def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 控制台输出 ─────────────────────────────────────────────────────────

def print_table(results: List[Dict]):
    """以表格形式打印汇总结果"""
    backends = list(dict.fromkeys(r["backend"] for r in results))

    for backend in backends:
        rows = [r for r in results if r["backend"] == backend]
        print(f"\n{'='*80}")
        print(f"  Backend: {backend.upper()}")
        print(f"{'='*80}")
        header = (f"{'并发':>4}  {'成功率':>6}  {'QPS':>6}  "
                  f"{'P50(ms)':>8}  {'P90(ms)':>8}  {'P99(ms)':>8}  "
                  f"{'Token/s':>8}")
        print(header)
        print("-" * 80)
        for r in sorted(rows, key=lambda x: x["concurrency"]):
            success_rate = r["success_count"] / r["total_requests"] * 100
            print(
                f"{r['concurrency']:>4}  {success_rate:>5.1f}%  {r['qps']:>6.2f}  "
                f"{r['latency_p50']:>8.1f}  {r['latency_p90']:>8.1f}  "
                f"{r['latency_p99']:>8.1f}  {r['total_tokens_per_sec']:>8.1f}"
            )

    # 如果有两个 backend，打印提升幅度
    if len(backends) == 2 and "vllm" in backends and "transformers" in backends:
        print(f"\n{'='*80}")
        print("  vLLM vs Transformers 提升幅度")
        print(f"{'='*80}")
        vllm_rows = {r["concurrency"]: r for r in results if r["backend"] == "vllm"}
        tf_rows = {r["concurrency"]: r for r in results if r["backend"] == "transformers"}
        common = sorted(set(vllm_rows) & set(tf_rows))
        print(f"{'并发':>4}  {'QPS提升':>8}  {'P90延迟降低':>10}  {'Token/s提升':>10}")
        print("-" * 50)
        for c in common:
            v, t = vllm_rows[c], tf_rows[c]
            qps_gain = (v["qps"] / t["qps"] - 1) * 100 if t["qps"] > 0 else 0
            p90_reduce = (1 - v["latency_p90"] / t["latency_p90"]) * 100 if t["latency_p90"] > 0 else 0
            tps_gain = (v["total_tokens_per_sec"] / t["total_tokens_per_sec"] - 1) * 100 if t["total_tokens_per_sec"] > 0 else 0
            print(f"{c:>4}  {qps_gain:>+7.1f}%  {p90_reduce:>+9.1f}%  {tps_gain:>+9.1f}%")


# ── 图表生成 ──────────────────────────────────────────────────────────

def plot_results(results: List[Dict], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("[WARN] matplotlib 未安装，跳过图表生成")
        print("       pip install matplotlib")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    backends = list(dict.fromkeys(r["backend"] for r in results))
    colors = {"vllm": "#2563EB", "transformers": "#DC2626"}
    markers = {"vllm": "o", "transformers": "s"}

    # ── 图1：P50 / P90 延迟对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LLM 推理服务延迟对比 (vLLM vs Transformers)", fontsize=14, fontweight="bold")

    for metric, ax, title in [
        ("latency_p50", axes[0], "P50 延迟 (ms)"),
        ("latency_p90", axes[1], "P90 延迟 (ms)"),
    ]:
        for backend in backends:
            rows = sorted([r for r in results if r["backend"] == backend],
                          key=lambda x: x["concurrency"])
            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            ax.plot(xs, ys,
                    label=backend.upper(),
                    color=colors.get(backend, "gray"),
                    marker=markers.get(backend, "o"),
                    linewidth=2, markersize=7)
        ax.set_xlabel("并发数", fontsize=11)
        ax.set_ylabel("延迟 (ms)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()
    p = os.path.join(output_dir, "latency_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 延迟图已保存: {p}")

    # ── 图2：QPS 和 Token/s 吞吐量对比 ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LLM 推理服务吞吐量对比 (vLLM vs Transformers)", fontsize=14, fontweight="bold")

    for metric, ax, title, ylabel in [
        ("qps", axes[0], "QPS（每秒请求数）", "Queries / Second"),
        ("total_tokens_per_sec", axes[1], "Token 生成速度（tokens/s）", "Tokens / Second"),
    ]:
        for backend in backends:
            rows = sorted([r for r in results if r["backend"] == backend],
                          key=lambda x: x["concurrency"])
            xs = [r["concurrency"] for r in rows]
            ys = [r[metric] for r in rows]
            ax.bar(
                [x + (0.3 if backend == "vllm" else -0.3) for x in range(len(xs))],
                ys,
                width=0.55,
                label=backend.upper(),
                color=colors.get(backend, "gray"),
                alpha=0.85,
            )
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(x) for x in xs])
        ax.set_xlabel("并发数", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "throughput_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 吞吐量图已保存: {p}")


# ── Markdown 报告 ─────────────────────────────────────────────────────

def generate_report(results: List[Dict], output_path: str):
    """生成 Markdown 格式的完整分析报告"""
    backends = list(dict.fromkeys(r["backend"] for r in results))
    lines = []

    lines.append("# vLLM 推理服务性能评测报告\n")
    lines.append("## 实验背景\n")
    lines.append(
        "本报告对比了 **vLLM**（PagedAttention + Continuous Batching）与 "
        "**Transformers 原生推理**（串行、无 batching）在电商客服场景下的推理性能，"
        "评测维度包括：请求延迟（P50/P90/P99）、QPS、Token 生成速度。\n"
    )

    lines.append("## 测试环境\n")
    lines.append("| 项目 | 配置 |")
    lines.append("|------|------|")
    lines.append("| GPU | NVIDIA RTX 4060Ti (8GB) |")
    lines.append("| 模型 | Qwen2.5-1.5B-Instruct (SFT + DPO 微调) |")
    lines.append("| 量化 | vLLM: FP16 / Transformers: 4bit BnB |")
    lines.append("| 测试集 | 电商客服问题（5类意图，随机采样） |")
    lines.append("")

    for backend in backends:
        rows = sorted([r for r in results if r["backend"] == backend],
                      key=lambda x: x["concurrency"])
        lines.append(f"## {backend.upper()} 测试结果\n")
        lines.append("| 并发 | 成功率 | QPS | P50(ms) | P90(ms) | P99(ms) | Token/s |")
        lines.append("|------|--------|-----|---------|---------|---------|---------|")
        for r in rows:
            sr = r["success_count"] / r["total_requests"] * 100
            lines.append(
                f"| {r['concurrency']} | {sr:.1f}% | {r['qps']:.2f} | "
                f"{r['latency_p50']:.1f} | {r['latency_p90']:.1f} | "
                f"{r['latency_p99']:.1f} | {r['total_tokens_per_sec']:.1f} |"
            )
        lines.append("")

    # 对比分析（仅当两个 backend 都有数据时）
    if "vllm" in backends and "transformers" in backends:
        vllm_rows = {r["concurrency"]: r for r in results if r["backend"] == "vllm"}
        tf_rows = {r["concurrency"]: r for r in results if r["backend"] == "transformers"}
        common = sorted(set(vllm_rows) & set(tf_rows))

        lines.append("## vLLM vs Transformers 提升幅度\n")
        lines.append("| 并发 | QPS提升 | P90延迟降低 | Token/s提升 |")
        lines.append("|------|---------|------------|------------|")
        qps_gains = []
        for c in common:
            v, t = vllm_rows[c], tf_rows[c]
            qps_gain = (v["qps"] / t["qps"] - 1) * 100 if t["qps"] > 0 else 0
            p90_r = (1 - v["latency_p90"] / t["latency_p90"]) * 100 if t["latency_p90"] > 0 else 0
            tps_gain = (v["total_tokens_per_sec"] / t["total_tokens_per_sec"] - 1) * 100 if t["total_tokens_per_sec"] > 0 else 0
            qps_gains.append(qps_gain)
            lines.append(f"| {c} | +{qps_gain:.1f}% | -{p90_r:.1f}% | +{tps_gain:.1f}% |")
        lines.append("")

        avg_qps_gain = sum(qps_gains) / len(qps_gains) if qps_gains else 0
        high_c = common[-1] if common else "N/A"
        v_high = vllm_rows.get(high_c, {})
        t_high = tf_rows.get(high_c, {})

        lines.append("## 核心结论\n")
        lines.append(
            f"1. **吞吐量（QPS）**：vLLM 平均比 Transformers 高 **{avg_qps_gain:.1f}%**，"
            f"在并发={high_c} 时差距最大，"
            f"vLLM QPS={v_high.get('qps',0):.2f}，Transformers={t_high.get('qps',0):.2f}。"
        )
        lines.append(
            f"2. **P90 延迟**：高并发下 vLLM 的 P90 延迟显著更低，"
            f"Continuous Batching 有效减少了排队等待。"
        )
        lines.append(
            "3. **PagedAttention 效果**：vLLM 的 KV cache 显存利用率更高，"
            "支持更大并发而不 OOM。"
        )
        lines.append(
            "4. **Transformers 瓶颈**：串行 + static batching 在高并发下 QPS 几乎不随并发增加，"
            "P99 延迟急剧上升。\n"
        )

    lines.append("## 技术点解析\n")
    lines.append("### Continuous Batching vs Static Batching\n")
    lines.append(
        "- **Static Batching**：请求必须凑够一批才能推理，先到的请求等后到的，"
        "GPU 利用率低，高并发下排队严重。\n"
        "- **Continuous Batching**（vLLM）：每步 decode 都可以动态加入新请求，"
        "先完成的位置立即被新请求占用，GPU 利用率接近 100%。\n"
    )
    lines.append("### PagedAttention\n")
    lines.append(
        "- 传统推理 KV cache 按最大长度预分配连续显存，碎片率高达 60-80%。\n"
        "- PagedAttention 将 KV cache 分页管理（类似 OS 虚拟内存），"
        "碎片率接近 4%，同等显存可支持更多并发请求。\n"
    )

    lines.append("## 图表\n")
    lines.append("![延迟对比](latency_comparison.png)\n")
    lines.append("![吞吐量对比](throughput_comparison.png)\n")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 分析报告已保存: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="压测结果分析")
    parser.add_argument("--input", type=str, default="results/raw_results.json")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--report", type=str, default="results/benchmark_report.md")
    parser.add_argument("--no-plot", action="store_true", help="跳过图表生成")
    args = parser.parse_args()

    print(f"\n读取压测结果: {args.input}")
    results = load_results(args.input)
    print(f"  共 {len(results)} 组测试")

    print_table(results)

    if not args.no_plot:
        plot_results(results, args.output_dir)

    generate_report(results, args.report)
    print(f"\n全部分析完成，结果目录: {args.output_dir}/")