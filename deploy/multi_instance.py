"""
多实例启动器：在单张GPU上起多个vLLM实例
用法：python deploy/multi_instance.py --model ./outputs/dpo_model --instances 2
"""

import subprocess
import argparse
import time
import sys
import signal
import requests

def wait_for_ready(port, timeout=120):
    """等待实例启动完成"""
    print(f"  等待端口 {port} 就绪...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                print(" ✓")
                return True
        except:
            pass
        time.sleep(2)
        print(".", end="", flush=True)
    print(" 超时！")
    return False

def start_instances(model_path, n_instances=2, base_port=8000):
    """启动多个vLLM实例"""
    # 每个实例分配的显存比例
    # 2个实例各35%，留30%给系统；3个实例各25%
    mem_per_instance = {2: 0.44, 3: 0.28}.get(n_instances, 0.35)
    
    processes = []
    ports = []

    for i in range(n_instances):
        port = base_port + i
        ports.append(port)
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(port),
            "--gpu-memory-utilization", str(mem_per_instance),
            "--max-model-len", "512",       # 调小以节省KV Cache显存
            "--served-model-name", "customer-service-llm",
        ]
        
        print(f"[实例{i+1}] 启动在端口 {port}，显存占比 {mem_per_instance}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # 不打印vLLM日志，保持终端干净
            stderr=subprocess.DEVNULL,
        )
        processes.append(proc)
        time.sleep(30)  # 错开启动时间，避免同时抢显存

    # 等待所有实例就绪
    print("\n等待所有实例启动...")
    all_ready = all(wait_for_ready(p) for p in ports)
    
    if not all_ready:
        print("部分实例启动失败，退出")
        for proc in processes:
            proc.terminate()
        sys.exit(1)
    
    print(f"\n✓ {n_instances}个实例全部就绪")
    print(f"  端口列表: {ports}")
    print(f"  现在可以启动NGINX并运行压测")
    print(f"\n按 Ctrl+C 停止所有实例\n")
    
    return processes, ports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/azir/projects/output/merged_model")
    parser.add_argument("--instances", type=int, default=2, choices=[2, 3])
    parser.add_argument("--base-port", type=int, default=8000)
    args = parser.parse_args()

    processes, ports = start_instances(
        args.model, args.instances, args.base_port
    )

    # 捕获Ctrl+C，优雅退出
    def shutdown(sig, frame):
        print("\n正在停止所有实例...")
        for proc in processes:
            proc.terminate()
        print("✓ 全部停止")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 保持运行
    for proc in processes:
        proc.wait()

if __name__ == "__main__":
    main()