"""
collect_outputs.py —— 质量评测第 1 步：采集各部署后端的回复

把冻结评测集 (config.EVAL_PROMPTS) 逐条发给指定后端，**贪心解码**采集回复，
落盘 {OUT_DIR}/answers_{tag}.jsonl，字段与微调项目 judge.py 期望的完全一致：
    {id, question, intent, bucket, answer}

为什么必须贪心 (temperature=0 + 固定 seed)：
    本项目质量轴只有一个变量——精度 (FP16 vs 4bit)。采样解码会引入随机性，
    把精度差异淹没在采样噪声里；贪心让输出确定，差异才能 100% 归因于精度。
    （这也是 README 2 里 fix#3「采样导致不可复现」的同一条纪律。）

为什么逐个后端跑：三组共用一张 8GB GPU、不能同开。标准流程：
    启动 A(:8000) → collect --tag fp16 → 关掉 → 启动 C(:8001) → collect --tag int4

用法：
    python quality/collect_outputs.py --tag fp16
    python quality/collect_outputs.py --tag int4
    python quality/collect_outputs.py --tag fp16 int4   # 若两个后端恰好都在线
"""
import argparse
import time

import config
from common import read_jsonl, write_jsonl
from openai import OpenAI


def collect_one(tag):
    cfg = config.MODELS[tag]
    cli = OpenAI(base_url=cfg["base_url"], api_key="EMPTY")   # 本地服务不校验 key
    prompts = read_jsonl(config.EVAL_PROMPTS)
    rows, t0 = [], time.time()

    for j, p in enumerate(prompts):
        ans = ""
        for attempt in range(3):                              # 轻量重试
            try:
                resp = cli.chat.completions.create(
                    model=cfg["model"],
                    messages=[{"role": "system", "content": config.SYSTEM_PROMPT},
                              {"role": "user",   "content": p["question"]}],
                    temperature=0, top_p=1, seed=config.SEED,
                    max_tokens=config.MAX_TOKENS, stream=False,
                )
                ans = (resp.choices[0].message.content or "").strip()
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [{tag}] id={p['id']} 失败：{e}")
                else:
                    time.sleep(2)

        rows.append({"id": p["id"], "question": p["question"],
                     "intent": p["intent"], "bucket": p["bucket"], "answer": ans})
        if (j + 1) % 20 == 0:
            print(f"  [{tag}] {j + 1}/{len(prompts)}  ({time.time() - t0:.0f}s)")

    out = f"{config.OUT_DIR}/answers_{tag}.jsonl"
    write_jsonl(out, rows)
    print(f"{tag} → {out} ({len(rows)} 条)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", nargs="+", required=True,
                    choices=list(config.MODELS.keys()))
    args = ap.parse_args()
    for tag in args.tag:
        collect_one(tag)
    print("\n采集完。下一步: export JUDGE_API_KEY=... && python quality/judge.py")


if __name__ == "__main__":
    main()
