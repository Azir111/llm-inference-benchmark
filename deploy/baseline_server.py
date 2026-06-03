"""
Transformers 原生推理服务（对照组）
使用 FastAPI + 单请求串行处理，无 continuous batching
用于与 vLLM 做性能对比

本版本新增：OpenAI-compatible 的 SSE 流式输出（stream=true），
使得可以与 vLLM 用同一套流式压测脚本测 TTFT / TPOT / ITL。

关键设计：流式生成期间 inference_lock 全程持有，直到整段回答生成完才释放。
这是 baseline 的核心——模拟“无 continuous batching”：并发请求只能串行排队，
后来的请求连 prefill 都要等前一个请求**整段**生成结束。
因此 baseline 的 TTFT 会随并发爆炸式增长（TTFT 里含了排队等待时间），
而 vLLM 的 Continuous Batching 能让 TTFT 保持平稳——这正是要展示的对比。
"""

import time
import json
import asyncio
import argparse
from threading import Thread
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)


app = FastAPI(title="Transformers Baseline Server")

# 全局模型和 tokenizer（单例）
model = None
tokenizer = None
# 用锁保证串行推理（模拟无 continuous batching）
inference_lock = asyncio.Lock()

# 迭代结束哨兵：用于把阻塞式 streamer 桥接到 asyncio 而不误吞 StopIteration
_STOP = object()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "customer-service-llm"
    messages: List[ChatMessage]
    max_tokens: int = 200
    temperature: float = 0.7
    stream: bool = False                       # 新增：是否流式
    stream_options: Optional[dict] = None      # 新增：支持 {"include_usage": true}


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
    return {"data": [{"id": "customer-service-llm", "object": "model"}]}


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def _sse(payload: dict) -> str:
    """格式化成一条 SSE 事件。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _next_or_stop(it):
    """在线程里取 streamer 的下一项；耗尽时返回哨兵而非抛 StopIteration。"""
    try:
        return next(it)
    except StopIteration:
        return _STOP


def _safe_generate(gen_kwargs: dict, streamer: TextIteratorStreamer):
    """在后台线程跑 generate；异常时强制结束 streamer，避免主协程死等。"""
    try:
        with torch.no_grad():
            model.generate(**gen_kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[generate error] {type(e).__name__}: {e}")
        try:
            streamer.end()
        except Exception:
            pass


def _build_inputs(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    return inputs


# ----------------------------------------------------------------------------
# 流式生成（核心）
# ----------------------------------------------------------------------------
async def _stream_chat(request: ChatRequest):
    created = int(time.time())
    chunk_id = "chatcmpl-baseline"
    include_usage = bool(request.stream_options and request.stream_options.get("include_usage"))

    def base_chunk(delta: dict, finish_reason=None):
        return {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    # 全程持锁 = 串行 = 无 batching。并发时后来的请求在这里排队，
    # 排队时间会算进它的 TTFT —— 这正是要展示的“TTFT 随并发爆炸”。
    async with inference_lock:
        inputs = _build_inputs(request)
        prompt_tokens = int(inputs["input_ids"].shape[1])

        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        # generate 必须在另一个线程跑，否则会阻塞事件循环、SSE 无法逐块下发
        thread = Thread(target=_safe_generate, args=(gen_kwargs, streamer), daemon=True)
        thread.start()

        # 首块：先吐 role（与 OpenAI 流式行为一致）
        yield _sse(base_chunk({"role": "assistant"}))

        full_text = ""
        it = iter(streamer)
        while True:
            # 用 to_thread 把阻塞式取 token 丢到线程池，期间让出事件循环 -> 逐块 flush
            piece = await asyncio.to_thread(_next_or_stop, it)
            if piece is _STOP:
                break
            if piece:
                full_text += piece
                yield _sse(base_chunk({"content": piece}))

        thread.join()

        # 结束块
        yield _sse(base_chunk({}, finish_reason="stop"))

        # usage 块（token 数用整段文本重新编码，比数 chunk 准）
        if include_usage:
            completion_tokens = len(
                tokenizer(full_text, add_special_tokens=False).input_ids
            )
            yield _sse({
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })

        yield "data: [DONE]\n\n"


# ----------------------------------------------------------------------------
# 端点：根据 stream 参数走流式或非流式
# ----------------------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # 流式：返回 SSE
    if request.stream:
        return StreamingResponse(
            _stream_chat(request),
            media_type="text/event-stream",
        )

    # 非流式：保留原有行为（老的端到端压测脚本仍可用）
    async with inference_lock:
        inputs = _build_inputs(request)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - t0

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
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(n_tokens),
            "total_tokens": int(inputs["input_ids"].shape[1] + n_tokens),
        },
        "_latency_s": round(latency, 3),
        "_tokens_per_sec": round(n_tokens / latency, 1) if latency > 0 else 0.0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformers 基线推理服务")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径，如 ~/projects/output/merged_model")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--no-4bit", action="store_true",
                        help="不使用 4bit 量化")
    args = parser.parse_args()

    load_model(args.model, use_4bit=not args.no_4bit)

    print(f"\n[OK] Transformers 基线服务启动于 http://0.0.0.0:{args.port}")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")