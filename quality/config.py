"""
质量评测配置 —— 复用自微调项目 eval/config.py 的同套约定，
改成压测项目的「精度」对照：A = vLLM FP16，C = Transformers 4bit。

裁判 (JUDGE_*) 必须比被测模型强、且不同家族（避免自我偏好）。
裁判 API key 从环境变量读取，不写进代码。
"""
import os

# ---------- 评测集 / 输出路径 ----------
EVAL_PROMPTS = "quality/eval_prompts.jsonl"   # prepare_eval_set.py 产出的「冻结」评测集
OUT_DIR      = "quality/outputs"              # answers_*.jsonl / pointwise / pairwise

# ---------- 采集（贪心解码，必须确定性）----------
SEED       = 42
MAX_TOKENS = 256          # 与压测 load_test_stream 的 max_tokens 对齐，便于交叉引用

# ⚠️ 必须与训练 / deploy/*.py 里逐字一致，否则训练-推理格式错配会压低质量分。
#    这是 README 2 里 fix#4「训练/推理格式一致性」的延续。
SYSTEM_PROMPT = "你是专业的电商客服助手，请礼貌、准确地回答用户问题。"  # TODO: 换成与训练完全相同的那句

# ---------- 被测后端（同一张 8GB GPU，逐个启动，不能同开）----------
# tag → 该后端的 OpenAI-compatible 地址 + served-model-name
MODELS = {
    "fp16": {"base_url": "http://localhost:8000/v1", "model": "customer-service-llm"},  # A 组 vLLM FP16
    "int4": {"base_url": "http://localhost:8001/v1", "model": "customer-service-llm"},  # C 组 Transformers 4bit
    # 可选 sanity check：B 组 vLLM 单流 FP16，权重与 A 完全相同，
    # 质量应≈A，用来证明「batching 不改输出质量」（呼应你 README 里并发=1 时 A≈B 那个 sanity check）。
    # "fp16_single": {"base_url": "http://localhost:8002/v1", "model": "customer-service-llm"},
}

# ---------- 成对胜负（headline 指标）----------
# (win_tag, lose_tag)：把 FP16 当参照，回答「4bit 有没有掉质量」。
PAIRWISE   = [("fp16", "int4")]
JUDGE_SWAP = True          # A/B 双向各判一次，方向不一致记平局，抵消位置偏差

# ---------- 裁判 ----------
JUDGE_MODEL    = "deepseek-chat"
JUDGE_BASE_URL = "https://api.deepseek.com"
JUDGE_API_KEY  = os.environ.get("JUDGE_API_KEY", "")

# ---------- 延迟（用于 latency × quality 散点）----------
# 默认值取自 results/benchmark_report_ablation.md（A=fp16, C=int4），重测后改这里即可。
LATENCY = {
    "fp16": {"tpot_ms": 15.0, "qps_c16": 12.91, "ttft_p90_c16_ms": 52.4},
    "int4": {"tpot_ms": 35.8, "qps_c16": 0.47,  "ttft_p90_c16_ms": 37684.0},
}
