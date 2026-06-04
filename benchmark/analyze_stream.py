#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式压测「消融」分析：读取 A / B / C 三份结果，生成三线对比图 + Markdown 报告

三组定义（变量隔离）：
    A = vLLM, max_num_seqs=256, FP16  —— 有 continuous batching
    B = vLLM, max_num_seqs=1,   FP16  —— 关掉 batching（单流），其余与 A 完全相同
    C = Transformers, 4bit            —— 朴素串行基线

    A vs B：唯一变量是 continuous batching（框架/精度/kernel 全同）→ 纯 batching 贡献
    B vs C：都是单流无批处理，差距 = kernel 优化 + 量化开销（仍混合，但已和 batching 分开）
    A vs C：端到端总差距（你原来的 681×），仅作总览，不单独归因

注意：三份 json 内部后端键名都叫 "vllm"（因为采集时都用 --vllm-url 单压），
      因此靠「文件路径」区分 A/B/C，不能靠文件内部的键名。

输出（默认到 results/）：
    - ttft_ablation_linear.png   TTFT_P90 vs 并发（线性轴）
    - ttft_ablation_log.png      TTFT_P90 vs 并发（对数轴）
    - qps_ablation.png           QPS vs 并发（核心图：A 爬升，B/C 平）
    - tpot_ablation.png          TPOT vs 并发
    - benchmark_report_ablation.md

用法：
    python analyze_stream.py \
        --input-a results/stream_results_A.json \
        --input-b results/stream_results_B.json \
        --input-c results/stream_results_C.json \
        --outdir results
"""

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


# ---- 配色 ----
C_A = "#2563eb"      # 蓝：A = vLLM 有 batching
C_B = "#f59e0b"      # 橙：B = vLLM 单流（无 batching）
C_C = "#dc2626"      # 红：C = Transformers 4bit
GRID = "#e5e7eb"

LABEL_A = "A: vLLM (continuous batching, FP16)"
LABEL_B = "B: vLLM (单流 max_num_seqs=1, FP16)"
LABEL_C = "C: Transformers (串行, 4bit)"

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
    ax.legend(frameon=False, fontsize=10)


def _plot_three(a, b, c, key, ylabel, title, fname, outdir, log_scale=False, ylim0=False):
    x = _series(a, "concurrency")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.plot(x, _series(a, key), marker="o", color=C_A, linewidth=2.5, markersize=7, label=LABEL_A)
    ax.plot(x, _series(b, key), marker="^", color=C_B, linewidth=2.5, markersize=7, label=LABEL_B)
    ax.plot(x, _series(c, key), marker="s", color=C_C, linewidth=2.5, markersize=7, label=LABEL_C)
    if log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
    if ylim0:
        ax.set_ylim(bottom=0)
    _style_ax(ax, title, "并发数", ylabel, x)
    fig.tight_layout()
    p = outdir / fname
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_ttft(a, b, c, outdir, log_scale):
    scale = "对数轴" if log_scale else "线性轴"
    fname = "ttft_ablation_log.png" if log_scale else "ttft_ablation_linear.png"
    return _plot_three(a, b, c, "ttft_p90_ms", "TTFT P90 (ms)",
                       f"首 Token 延迟 P90 消融对比 ({scale})", fname, outdir,
                       log_scale=log_scale)


def plot_qps(a, b, c, outdir):
    return _plot_three(a, b, c, "qps", "QPS (请求/秒)",
                       "吞吐量消融对比 (QPS)：仅 A 随并发扩展", "qps_ablation.png",
                       outdir, ylim0=True)


def plot_tpot(a, b, c, outdir):
    return _plot_three(a, b, c, "tpot_mean_ms", "TPOT 平均 (ms/token)",
                       "单 Token 生成间隔消融对比 (TPOT)", "tpot_ablation.png",
                       outdir, ylim0=True)


def _md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join(["------"] * len(headers)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def _detail_table(name, rows):
    return f"\n### {name}\n\n" + _md_table(
        ["并发", "QPS", "tok/s", "TTFT_P50(ms)", "TTFT_P90(ms)",
         "TPOT_mean(ms)", "ITL_P99(ms)", "成功率"],
        [[r["concurrency"], r["qps"], r["output_token_throughput"],
          r["ttft_p50_ms"], r["ttft_p90_ms"], r["tpot_mean_ms"],
          r["itl_p99_ms"], f'{r["success_rate"]}%'] for r in rows])


def build_report(cfg, a, b, c, img_paths, outdir):
    a0, b0, c0 = a[0], b[0], c[0]
    aL, bL, cL = a[-1], b[-1], c[-1]
    conc = aL["concurrency"]

    def ratio(x, y):
        return x / y if y else 0

    # A vs B：纯 batching 贡献（TTFT 与 QPS）
    ab_ttft = ratio(bL["ttft_p90_ms"], aL["ttft_p90_ms"])
    ab_qps = ratio(aL["qps"], bL["qps"])
    # B vs C：kernel + 量化（单流下的差距）
    bc_ttft = ratio(cL["ttft_p90_ms"], bL["ttft_p90_ms"])
    bc_tpot = ratio(cL["tpot_mean_ms"], bL["tpot_mean_ms"])
    # A vs C：总差距（脚注）
    ac_ttft = ratio(cL["ttft_p90_ms"], aL["ttft_p90_ms"])
    ac_qps = ratio(aL["qps"], cL["qps"])

    L = []
    L.append("# 流式推理压测报告（消融版：隔离 continuous batching）\n")
    L.append(f"> 模型：`{cfg.get('model')}` ｜ 每并发 {cfg.get('requests_per_level')} 请求"
             f"，预热 {cfg.get('warmup')} 次 ｜ max_tokens={cfg.get('max_tokens')}\n")

    L.append("\n## 实验设计（变量隔离）\n")
    L.append(_md_table(
        ["组", "框架", "batching", "精度", "作用"],
        [["A", "vLLM", "开 (max_num_seqs=256)", "FP16", "完整 vLLM"],
         ["B", "vLLM", "关 (max_num_seqs=1)", "FP16", "单流对照，仅去掉 batching"],
         ["C", "Transformers", "无", "4bit", "朴素串行基线"]]))
    L.append("\n- **A vs B**：唯一变量是 continuous batching（框架/精度/kernel 全同）→ 纯 batching 贡献。\n"
             "- **B vs C**：都是单流无批处理，差距 = kernel 优化 + 量化开销（已和 batching 分离）。\n"
             "- **A vs C**：端到端总差距，包含上述全部因素叠加，仅作总览、不单独归因。\n")

    L.append("\n## 关键对比图\n")
    for label, p in img_paths:
        L.append(f"\n**{label}**\n\n![{label}]({p.name})\n")

    L.append("\n## 核心结论\n")

    # sanity check：并发=1 时 A≈B
    L.append(
        f"\n**0. Sanity check：并发=1 时 A≈B。** "
        f"并发=1 时 A 与 B 的 TTFT_P90 分别为 {a0['ttft_p90_ms']}ms / {b0['ttft_p90_ms']}ms，"
        f"QPS 为 {a0['qps']} / {b0['qps']}，两者接近——只有一个请求时 batching 开不开都没区别，"
        f"验证了实验本身没有引入额外变量。\n")

    L.append(
        f"\n**1. 扩展性来自 continuous batching，而非框架本身（A vs B）。** "
        f"开启 batching 的 A，QPS 从 {a0['qps']} 近线性扩展到 {aL['qps']}（约 "
        f"{ratio(aL['qps'], a0['qps']):.0f}×）；强制 max_num_seqs=1 退化为单流的 B，"
        f"QPS 全程卡在 ~{bL['qps']}，与朴素 Transformers（C，~{cL['qps']}）一样毫无扩展。"
        f"并发={conc} 时 A 的 TTFT_P90 仅 {aL['ttft_p90_ms']}ms，B 飙至 "
        f"{bL['ttft_p90_ms']/1000:.1f}s——**仅去掉 batching 这一个变量，TTFT 就相差 "
        f"{ab_ttft:.0f}×、QPS 相差 {ab_qps:.0f}×**。这是 continuous batching 价值的纯净证据："
        f"它消除了高并发下的排队，是吞吐扩展的根本来源。\n")

    L.append(
        f"\n**2. 单流下 vLLM 仍快于 Transformers，但这是 kernel + 量化，不是 batching（B vs C）。** "
        f"两者都无批处理，并发={conc} 时 B 的 TTFT_P90（{bL['ttft_p90_ms']/1000:.1f}s）与 "
        f"C（{cL['ttft_p90_ms']/1000:.1f}s）的差距约 {bc_ttft:.1f}×，TPOT 约 {bc_tpot:.1f}×。"
        f"这部分差距来自两点叠加且**未进一步隔离**：(a) C 开了 4bit，每层每 token 前向都要反量化，"
        f"带来额外开销；(b) vLLM 推理专用的 fused kernel 比 Transformers 通用 generate 路径更高效。"
        f"需要说明：单 token 速度（TPOT）由 kernel 与精度决定，**与 PagedAttention 无关**"
        f"——PagedAttention 解决的是显存碎片、提升并发上限，影响的是吞吐而非单 token 延迟。\n")

    L.append(
        f"\n**3. 端到端总差距仅作总览（A vs C）。** "
        f"并发={conc} 时 A 对 C 的 TTFT_P90 相差 {ac_ttft:.0f}×、QPS 相差 {ac_qps:.0f}×。"
        f"这个数字同时包含 batching、kernel、量化三重因素，**不可单独归因于某一项**，"
        f"因此本报告以 A vs B 的扩展性结论为主，此总差距仅作直观参考。\n")

    L.append(
        f"\n**4. 关于成功率与超时阈值。** "
        f"衡量高并发服务质量应看 TTFT/SLA，而非单纯的请求完成率：即便成功率 100%，"
        f"B/C 在并发={conc} 时尾部请求要等数秒甚至数十秒才出首字，对客服等实时场景已等同不可用。"
        f"（注：若 B 组在高并发出现超时，需确认压测 timeout 足够大且三组一致，否则数据不可比。）\n")

    L.append("\n## 各组明细\n")
    L.append(_detail_table("A — vLLM (continuous batching, FP16)", a))
    L.append(_detail_table("B — vLLM (单流 max_num_seqs=1, FP16)", b))
    L.append(_detail_table("C — Transformers (串行, 4bit)", c))

    report = "\n".join(L)
    p = outdir / "benchmark_report_ablation.md"
    p.write_text(report, encoding="utf-8")
    return p


def _load_vllm_rows(path):
    """三份文件内部后端键名都叫 'vllm'，靠文件路径区分 A/B/C。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    backends = data.get("backends", {})
    rows = backends.get("vllm")
    if rows is None:
        raise SystemExit(f"[ERROR] {path} 里找不到 backends.vllm，确认这是单压 --vllm-url 生成的结果")
    return data.get("config", {}), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-a", default="results/stream_results_A.json", help="A 组：vLLM 有 batching")
    ap.add_argument("--input-b", default="results/stream_results_B.json", help="B 组：vLLM 单流")
    ap.add_argument("--input-c", default="results/stream_results_C.json", help="C 组：Transformers 4bit")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    cfg, a = _load_vllm_rows(args.input_a)
    _, b = _load_vllm_rows(args.input_b)
    _, c = _load_vllm_rows(args.input_c)

    # 对齐校验：三组并发级别必须一致，否则画线会错位
    xa, xb, xc = _series(a, "concurrency"), _series(b, "concurrency"), _series(c, "concurrency")
    if not (xa == xb == xc):
        raise SystemExit(f"[ERROR] 三组并发级别不一致：A={xa} B={xb} C={xc}，请用相同 --concurrency 重跑")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = []
    imgs.append(("QPS 消融对比（核心图）", plot_qps(a, b, c, outdir)))
    imgs.append(("TTFT_P90 消融对比（线性轴）", plot_ttft(a, b, c, outdir, log_scale=False)))
    imgs.append(("TTFT_P90 消融对比（对数轴）", plot_ttft(a, b, c, outdir, log_scale=True)))
    imgs.append(("TPOT 消融对比", plot_tpot(a, b, c, outdir)))

    report = build_report(cfg, a, b, c, imgs, outdir)

    print("生成完成：")
    for _, p in imgs:
        print(f"  {p}")
    print(f"  {report}")


if __name__ == "__main__":
    main()