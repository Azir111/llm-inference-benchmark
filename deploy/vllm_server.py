"""
vLLM 高并发推理服务
基于 OpenAI-compatible API，支持 continuous batching

新增（用于消融实验）：
  --max-num-seqs  一个 batch 最多并发序列数；设为 1 可退化为无 continuous batching
  --dtype         计算精度；做对照实验时务必两组显式写同一个值，锁死精度变量
"""

import argparse
import subprocess
import sys
import time
import requests
import json
from pathlib import Path


def check_vllm_installed():
    try:
        import vllm
        print(f"[OK] vLLM 已安装: {vllm.__version__}")
        return True
    except ImportError:
        print("[ERROR] vLLM 未安装，请运行: pip install vllm")
        return False


def start_server(
    model_path: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 2048,
    quantization: str = None,
    tensor_parallel_size: int = 1,
    max_num_seqs: int = 256,
    dtype: str = "auto",
):
    """
    启动 vLLM OpenAI-compatible 服务

    Args:
        model_path: 模型路径（本地路径或 HuggingFace model id）
        port: 服务端口
        gpu_memory_utilization: GPU 显存占用比例（0-1）
        max_model_len: 最大序列长度
        quantization: 量化方式 (awq / gptq / None)
        tensor_parallel_size: 张量并行数（多卡时使用）
        max_num_seqs: 一个 batch 最多并发序列数（默认 256）；
                      设为 1 退化为单流，等价于关闭 continuous batching（消融实验用）
        dtype: 计算精度 (auto / float16 / bfloat16 / float32)；
               做 A/B 对照时两组务必显式传同一个值，避免精度变量混入
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--max-model-len", str(max_model_len),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--served-model-name", "customer-service-llm",
        # 关键：continuous batching 由 max_num_seqs 控制（>1 即开启，=1 退化为单流）
        "--max-num-seqs", str(max_num_seqs),
        "--dtype", dtype,
    ]

    if quantization:
        cmd.extend(["--quantization", quantization])

    print("\n" + "="*60)
    print("启动 vLLM 服务")
    print("="*60)
    print(f"  模型路径:     {model_path}")
    print(f"  端口:         {port}")
    print(f"  显存占用:     {gpu_memory_utilization*100:.0f}%")
    print(f"  最大序列长:   {max_model_len}")
    print(f"  量化:         {quantization or '无'}")
    print(f"  张量并行:     {tensor_parallel_size}")
    print(f"  max_num_seqs: {max_num_seqs}{'  (单流 / 无 continuous batching)' if max_num_seqs == 1 else ''}")
    print(f"  dtype:        {dtype}")
    print("="*60)
    print(f"\n启动命令:\n  {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # 等待服务就绪
    print("等待服务启动...")
    max_wait = 120  # 最多等 120 秒
    start = time.time()
    while time.time() - start < max_wait:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                print(f"[OK] 服务已就绪，耗时 {time.time()-start:.1f}s")
                return process
        except Exception:
            pass
        # 打印启动日志
        line = process.stdout.readline()
        if line:
            print(f"  {line.rstrip()}")
        time.sleep(1)

    print("[ERROR] 服务启动超时")
    process.terminate()
    return None


def test_server(port: int = 8000):
    """发送一条测试请求，验证服务正常"""
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "customer-service-llm",
        "messages": [
            {"role": "user", "content": "我的订单什么时候发货？"}
        ],
        "max_tokens": 200,
        "temperature": 0.7,
    }
    print("\n发送测试请求...")
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=30)
    latency = (time.time() - t0) * 1000
    data = r.json()
    answer = data["choices"][0]["message"]["content"]
    print(f"[OK] 响应延迟: {latency:.0f}ms")
    print(f"  回复: {answer[:100]}...")


def show_model_info(port: int = 8000):
    """展示已加载的模型信息"""
    r = requests.get(f"http://localhost:{port}/v1/models")
    models = r.json()
    print("\n已加载模型:")
    for m in models.get("data", []):
        print(f"  - {m['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM 推理服务启动器")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径，如 ./outputs/sft_model 或 Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--quantization", type=str, default=None,
                        choices=["awq", "gptq", None])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="一个 batch 最多并发序列数；设为 1 可退化为无 continuous batching（消融实验用）")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="计算精度；做 A/B 对照时两组务必显式传同一个值，锁死精度变量")
    parser.add_argument("--test-only", action="store_true",
                        help="只发测试请求，不启动服务（服务已在运行时使用）")
    args = parser.parse_args()

    if not check_vllm_installed():
        sys.exit(1)

    if args.test_only:
        test_server(args.port)
        show_model_info(args.port)
    else:
        proc = start_server(
            model_path=args.model,
            port=args.port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.max_num_seqs,
            dtype=args.dtype,
        )
        if proc:
            test_server(args.port)
            show_model_info(args.port)
            print(f"\n服务运行中，按 Ctrl+C 停止")
            try:
                proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                print("\n服务已停止")