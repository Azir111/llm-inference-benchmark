"""
Transformers 原生推理服务（对照组）
使用 FastAPI + 单请求串行处理，无 continuous batching
用于与 vLLM 做性能对比
"""

import time
import asyncio
import argparse
from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


app = FastAPI(title="Transformers Baseline Server")

# 全局模型和 tokenizer（单例）
model = None
tokenizer = None
# 用锁保证串行推理（模拟无 continuous batching）
inference_lock = asyncio.Lock()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "customer-service-llm"
    messages: List[ChatMessage]
    max_tokens: int = 200
    temperature: float = 0.7


class ChatResponse(BaseModel):
    id: str = "chatcmpl-baseline"
    object: str = "chat.completion"
    model: str = "customer-service-llm"
    choices: list
    usage: dict


def load_model(model_path: str, use_4bit: bool = True):
    global model, tokenizer
    print(f"[加载模型] {model_path}")
    print(f"  4bit量化: {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    print("[OK] 模型加载完成")


@app.get("/health")
def health():
    return {"status": "ok", "backend": "transformers"}


@app.get("/v1/models")
def list_models():
    return {
        "data": [{"id": "customer-service-llm", "object": "model"}]
    }


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    """
    串行推理端点
    注意：这里用 asyncio.Lock 模拟无 batching 的串行处理
    """
    async with inference_lock:
        # 构建 prompt
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        # 应用 chat template（兼容 Qwen2.5）
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        # 只取新生成的 tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        n_tokens = len(new_tokens)

    return {
        "id": "chatcmpl-baseline",
        "object": "chat.completion",
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": n_tokens,
            "total_tokens": inputs["input_ids"].shape[1] + n_tokens,
        },
        # 附加性能信息（非标准字段，便于分析）
        "_latency_s": round(latency, 3),
        "_tokens_per_sec": round(n_tokens / latency, 1),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers 基线推理服务")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径，如 ./outputs/sft_model")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-4bit", action="store_true",
                        help="不使用 4bit 量化")
    args = parser.parse_args()

    load_model(args.model, use_4bit=not args.no_4bit)

    print(f"\n[OK] Transformers 基线服务启动于 http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")