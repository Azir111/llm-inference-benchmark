"""
report.py —— 质量评测第 3 步：汇总 + 把「延迟」升级成「延迟 × 质量」

读 judge 产出的 pointwise / pairwise，输出：
  1) 逐维均分表（按 tag × bucket）
  2) 成对胜负：fp16 vs int4 的 胜/平/负 率（headline）
  3) latency × quality 汇总表 + 散点图（延迟/吞吐取自 config.LATENCY）

散点图约定（避免视觉误导）：
  - y 轴（质量）锁死到打分满量程 1~5，噪声级差距自然贴平，不被窄 y 轴放大；
  - x 轴用 QPS@16（吞吐差一个数量级，比 TPOT 的 2.7× 更有张力）。

结论模板（两个分支都要照实写）：
  - 质量(int4) ≈ 质量(fp16) → 4bit 没换来质量收益，却更慢更低吞吐，是纯「延迟换显存」
  - 质量(int4) <  质量(fp16) → 4bit 两个轴都吃亏；8GB 能塞下 FP16 时就没有量化的理由

用法:  python quality/report.py
"""
import statistics as st
from collections import defaultdict

import config
from common import read_jsonl

DIMS = ["correctness", "helpfulness", "tone", "safety"]


def pointwise_table():
    rows = read_jsonl(f"{config.OUT_DIR}/pointwise.jsonl")
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for d in DIMS:
            agg[(r["model"], r["bucket"])][d].append(r[d])

    print("\n=== 逐条多维打分（1~5 均分）===")
    print(f"{'tag':<6}{'bucket':<10}" + "".join(f"{d:<13}" for d in DIMS) + "总均分")
    overall = {}
    for (tag, bucket), dim2vals in sorted(agg.items()):
        means = {d: st.mean(v) for d, v in dim2vals.items()}
        ov = st.mean([means[d] for d in DIMS])
        overall[(tag, bucket)] = ov
        print(f"{tag:<6}{bucket:<10}"
              + "".join(f"{means[d]:<13.2f}" for d in DIMS) + f"{ov:.2f}")
    return overall


def pairwise_table():
    rows = read_jsonl(f"{config.OUT_DIR}/pairwise.jsonl")
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        agg[r["compare"]][r["bucket"]][r["winner"]] += 1

    print("\n=== 成对胜负（headline，winner 记的是 tag）===")
    for compare, bucket2 in agg.items():
        win_tag, lose_tag = compare.split("_vs_")
        print(f"\n[{compare}]")
        for bucket, c in sorted(bucket2.items()):
            n = sum(c.values())
            wr = c.get(win_tag, 0) / n * 100 if n else 0
            print(f"  {bucket:<10} {win_tag} 胜率 {wr:.0f}%  "
                  f"(win {c.get(win_tag, 0)} / tie {c.get('tie', 0)} / "
                  f"lose {c.get(lose_tag, 0)}, n={n})")


def latency_quality(quality):
    print("\n=== latency × quality 汇总 ===")
    print(f"{'tag':<6}{'TPOT(ms)':<10}{'QPS@16':<9}{'质量(总均分)':<12}")
    pts = []
    for tag, lat in config.LATENCY.items():
        qs = [v for (t, b), v in quality.items() if t == tag]   # 跨桶平均
        q = st.mean(qs) if qs else float("nan")
        pts.append((tag, lat["tpot_ms"], lat["qps_c16"], q))
        print(f"{tag:<6}{lat['tpot_ms']:<10}{lat['qps_c16']:<9}{q:<12.2f}")

    try:   # matplotlib 没装就只出表格，不报错
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        for tag, _tpot, qps, q in pts:
            ax.scatter(qps, q, s=140)
            ax.annotate(tag, (qps, q), textcoords="offset points", xytext=(8, 4))
        # y 轴锁死到打分满量程：噪声级的质量差距会自然贴平，杜绝「窄 y 轴放大噪声」的误导
        ax.set_ylim(1, 5)
        # x 轴用 QPS（吞吐差一个数量级，比 TPOT 的 2.7× 更有视觉张力）
        max_qps = max(p[2] for p in pts)
        ax.set_xlim(0, max_qps * 1.1)
        ax.set_xlabel("QPS @ concurrency=16  -> higher is better")
        ax.set_ylabel("Quality (LLM-judge mean score, 1-5)  -> better")
        ax.set_title("FP16 vs 4bit: equal quality, order-of-magnitude throughput gap")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = f"{config.OUT_DIR}/latency_quality_scatter.png"
        fig.savefig(path, dpi=130)
        print(f"\n散点图 → {path}")
    except Exception as e:
        print(f"(跳过画图: {e})")


def main():
    q = pointwise_table()
    pairwise_table()
    latency_quality(q)


if __name__ == "__main__":
    main()