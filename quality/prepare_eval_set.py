"""
prepare_eval_set.py —— 冻结质量评测集（只跑一次）

从原始数据（template + hard 两桶）产出固定的 quality/eval_prompts.jsonl：
  - hard 桶：全部保留（28 条手写非模板探针，是泛化/越界压力测试的核心）
  - template 桶：按 intent 分层抽样到 --per-intent 条（模板高度冗余，全测意义不大）

「冻结」很关键：采集阶段所有后端必须吃同一批、同顺序的 prompt，
FP16 vs 4bit 的质量对比才干净。所以这里 seed 固定、产物落盘一次后就别再动。

用法：
  python quality/prepare_eval_set.py --src raw_prompts.jsonl --per-intent 8
"""
import argparse
import random
from collections import defaultdict

import config
from common import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="原始 prompts jsonl（含 template + hard）")
    ap.add_argument("--per-intent", type=int, default=8,
                    help="每个 intent 保留几条 template；想全测就设很大")
    args = ap.parse_args()

    rows = read_jsonl(args.src)
    hard = [r for r in rows if r.get("bucket") == "hard"]
    tmpl = [r for r in rows if r.get("bucket") == "template"]

    by_intent = defaultdict(list)
    for r in tmpl:
        by_intent[r["intent"]].append(r)

    rng = random.Random(config.SEED)
    sampled = []
    for intent, items in sorted(by_intent.items()):
        rng.shuffle(items)
        sampled += items[:args.per_intent]

    # 只保留下游需要的字段（reference 不参与 judge，丢弃）
    out = [{"id": r["id"], "bucket": r["bucket"],
            "intent": r["intent"], "question": r["question"]}
           for r in hard + sampled]
    write_jsonl(config.EVAL_PROMPTS, out)
    print(f"冻结评测集 → {config.EVAL_PROMPTS}："
          f"hard {len(hard)} + template {len(sampled)} = {len(out)} 条")


if __name__ == "__main__":
    main()
