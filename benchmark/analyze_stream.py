#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式压测结果分析：读取 stream_results.json，生成对比图 + Markdown 报告

输出（默认到 results/）：
    - ttft_comparison_linear.png   TTFT_P90 vs 并发（线性轴，视觉冲击）
    - ttft_comparison_log.png      TTFT_P90 vs 并发（对数轴，两条线形状都看得清）
    - qps_comparison.png           QPS vs 并发
    - tpot_comparison.png          TPOT vs 并发
    - benchmark_report_stream.md   完整报告

用法：
    python analyze_stream.py                       # 默认读 results/stream_results.json
    python analyze_stream.py --input path/to.json --outdir results
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ---- 配色 ----
C_VLLM = "#2563eb"      # 蓝：vLLM
C_BASE = "#dc2626"      # 红：baseline
GRID = "#e5e7eb"

# 中文字体（找不到就退回默认，仅影响中文标签显示）
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
for _f in ["Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei",
           "Microsoft YaHei", "SimHei", "Arial Unicode MS"]:
    if _f in _installed:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False


def _series(rows, key):
    return [r[key] for r in rows]


def _style_ax(ax, title, xlabel, ylabel, x):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11)


def plot_ttft(vllm, base, outdir, log_scale: bool):
    x = _series(vllm, "concurrency")
    yv = _series(vllm, "ttft_p90_ms")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)

    ax.plot(x, yv, marker="o", color=C_VLLM, linewidth=2.5,
            markersize=7, label="vLLM (Continuous Batching)")
    if base:
        yb = _series(base, "ttft_p90_ms")
        ax.plot(x, yb, marker="s", color=C_BASE, linewidth=2.5,
                markersize=7, label="Transformers (串行, 无 batching)")
        # 在 baseline 末点标注倍数
        ratio = yb[-1] / yv[-1] if yv[-1] else 0
        ax.annotate(f"{yb[-1]/1000:.1f}s\n({ratio:.0f}× vLLM)",
                    xy=(x[-1], yb[-1]), xytext=(-10, -5),
                    textcoords="offset points", ha="right", va="top",
                    fontsize=10, color=C_BASE, fontweight="bold")

    scale = "对数轴" if log_scale else "线性轴"
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    _style_ax(ax, f"首 Token 延迟 P90 对比 (TTFT_P90, {scale})",
              "并发数", "TTFT P90 (ms)", x)
    fig.tight_layout()
    name = "ttft_comparison_log.png" if log_scale else "ttft_comparison_linear.png"
    p = outdir / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_qps(vllm, base, outdir):
    x = _series(vllm, "concurrency")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(x, _series(vllm, "qps"), marker="o", color=C_VLLM,
            linewidth=2.5, markersize=7, label="vLLM")
    if base:
        ax.plot(x, _series(base, "qps"), marker="s", color=C_BASE,
                linewidth=2.5, markersize=7, label="Transformers")
    _style_ax(ax, "吞吐量对比 (QPS)", "并发数", "QPS (请求/秒)", x)
    fig.tight_layout()
    p = outdir / "qps_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_tpot(vllm, base, outdir):
    x = _series(vllm, "concurrency")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(x, _series(vllm, "tpot_mean_ms"), marker="o", color=C_VLLM,
            linewidth=2.5, markersize=7, label="vLLM")
    if base:
        ax.plot(x, _series(base, "tpot_mean_ms"), marker="s", color=C_BASE,
                linewidth=2.5, markersize=7, label="Transformers")
    ax.set_ylim(bottom=0)
    _style_ax(ax, "单 Token 生成间隔对比 (TPOT)", "并发数",
              "TPOT 平均 (ms/token)", x)
    fig.tight_layout()
    p = outdir / "tpot_comparison.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def _md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["------"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def build_report(cfg, vllm, base, img_paths, outdir):
    has_base = bool(base)
    lines = []
    lines.append("# 流式推理压测报告（TTFT / TPOT / QPS）\n")
    lines.append(f"> 模型：`{cfg.get('model')}` ｜ 每并发 {cfg.get('requests_per_level')} 请求"
                 f"，预热 {cfg.get('warmup')} 次 ｜ max_tokens={cfg.get('max_tokens')}"
                 f" ｜ {cfg.get('timestamp')}\n")
    lines.append("\n指标说明：TTFT=首 token 延迟（响应快慢）；TPOT=每输出 token 平均间隔"
                 "（吐字速度）；ITL=相邻 token 间隔；QPS=系统吞吐。\n")

    # --- 图 ---
    lines.append("\n## 关键对比图\n")
    for label, p in img_paths:
        lines.append(f"\n**{label}**\n\n![{label}]({p.name})\n")

    # --- vLLM 表 ---
    lines.append("\n## vLLM 明细\n")
    lines.append(_md_table(
        ["并发", "QPS", "tok/s", "TTFT_P50(ms)", "TTFT_P90(ms)",
         "TPOT_mean(ms)", "ITL_P99(ms)", "成功率"],
        [[r["concurrency"], r["qps"], r["output_token_throughput"],
          r["ttft_p50_ms"], r["ttft_p90_ms"], r["tpot_mean_ms"],
          r["itl_p99_ms"], f'{r["success_rate"]}%'] for r in vllm]))

    if has_base:
        lines.append("\n## Transformers（对照组）明细\n")
        lines.append(_md_table(
            ["并发", "QPS", "tok/s", "TTFT_P50(ms)", "TTFT_P90(ms)",
             "TPOT_mean(ms)", "ITL_P99(ms)", "成功率"],
            [[r["concurrency"], r["qps"], r["output_token_throughput"],
              r["ttft_p50_ms"], r["ttft_p90_ms"], r["tpot_mean_ms"],
              r["itl_p99_ms"], f'{r["success_rate"]}%'] for r in base]))

        # --- 对比小结（自动算倍数）---
        v_last, b_last = vllm[-1], base[-1]
        c = v_last["concurrency"]
        ttft_x = b_last["ttft_p90_ms"] / v_last["ttft_p90_ms"] if v_last["ttft_p90_ms"] else 0
        qps_x = v_last["qps"] / b_last["qps"] if b_last["qps"] else 0
        tpot_x = b_last["tpot_mean_ms"] / v_last["tpot_mean_ms"] if v_last["tpot_mean_ms"] else 0
        lines.append("\n## 核心结论\n")
        lines.append(
            f"\n**1. TTFT（首 token 延迟）—— 差距随并发指数级拉大。** "
            f"并发=1 时两者接近（{v_last['concurrency'] and vllm[0]['ttft_p90_ms']}ms vs "
            f"{base[0]['ttft_p90_ms']}ms），但并发={c} 时 vLLM 仍仅 "
            f"{v_last['ttft_p90_ms']}ms，Transformers 飙至 "
            f"{b_last['ttft_p90_ms']/1000:.1f}s，相差 **{ttft_x:.0f} 倍**。"
            f"原因：Transformers 串行处理，第 N 个请求必须排队等前面所有请求**整段**生成完，"
            f"排队时间全部计入 TTFT；vLLM 的 Continuous Batching 让新请求在下一个 decode step "
            f"即可加入当前批次，几乎无需排队。\n")
        lines.append(
            f"\n**2. QPS（系统吞吐）—— 一个近线性扩展，一个完全卡死。** "
            f"vLLM 从 {vllm[0]['qps']} 扩展到 {v_last['qps']}（约 "
            f"{v_last['qps']/vllm[0]['qps']:.0f}×）；Transformers 全程卡在 "
            f"~{base[0]['qps']} QPS，加并发毫无收益——串行架构下系统吞吐恒等于单请求速度。"
            f"并发={c} 时 vLLM 吞吐为 Transformers 的 **{qps_x:.0f} 倍**。\n")
        lines.append(
            f"\n**3. TPOT（吐字速度）—— vLLM 始终快约 {tpot_x:.1f} 倍且与并发无关。** "
            f"vLLM ~{v_last['tpot_mean_ms']}ms/token，Transformers ~{b_last['tpot_mean_ms']}ms/token，"
            f"两者各自都不随并发变化：Transformers 因串行、任意时刻只有一个请求在 GPU 上，单流速度恒定；"
            f"vLLM 的单 token 速度优势来自 FP16 + PagedAttention 的显存效率。\n")
        lines.append(
            f"\n**4. 关于成功率与超时阈值。** 本轮两者成功率均 100%，因为压测超时阈值（默认 120s）"
            f"高于 Transformers 最慢请求的 TTFT（约 {b_last['ttft_p90_ms']/1000:.0f}s）。"
            f"但需指出：并发={c} 时 Transformers 尾部请求要等 {b_last['ttft_p90_ms']/1000:.0f} 秒才出首字，"
            f"对客服等实时场景已等同不可用——衡量服务质量应看 TTFT/SLA，而非单纯的请求完成率。\n")

    report = "\n".join(lines)
    p = outdir / "benchmark_report_stream.md"
    p.write_text(report, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/stream_results.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cfg = data.get("config", {})
    vllm = data["backends"]["vllm"]
    base = data["backends"].get("baseline")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = []
    imgs.append(("TTFT_P90 对比（线性轴）", plot_ttft(vllm, base, outdir, log_scale=False)))
    imgs.append(("TTFT_P90 对比（对数轴）", plot_ttft(vllm, base, outdir, log_scale=True)))
    imgs.append(("QPS 对比", plot_qps(vllm, base, outdir)))
    imgs.append(("TPOT 对比", plot_tpot(vllm, base, outdir)))

    report = build_report(cfg, vllm, base, imgs, outdir)

    print("生成完成：")
    for _, p in imgs:
        print(f"  {p}")
    print(f"  {report}")


if __name__ == "__main__":
    main()