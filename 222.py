import time
import numpy as np
import psutil
# 需在 Jetson 上运行: pip install jetson-stats
from jtop import jtop


class VLABenchmark:
    def __init__(self, model):
        self.model = model
        self.latencies = []

    def run_benchmark(self, sample_image, instruction, num_runs=100, warmup=10):
        print("Starting warmup...")
        for _ in range(warmup):
            _ = self.model.predict(sample_image, instruction)

        print("Starting benchmark...")
        with jtop() as jetson:
            for i in range(num_runs):
                start_time = time.time()

                # 执行模型推理
                action = self.model.predict(sample_image, instruction)

                end_time = time.time()
                latency = end_time - start_time
                self.latencies.append(latency)

                # 记录资源占用 (需在 jetson-stats 上下文中)
                if jetson.ok():
                    gpu_util = jetson.stats['GPU']
                    ram_usage = jetson.memory['RAM']['used'] / 1024  # 转换为 MB
                    cpu_util = jetson.stats['CPU1']  # 示例：读取 CPU1 占用率

                    if i % 10 == 0:
                        print(f"Run {i}: Latency={latency:.4f}s, GPU={gpu_util}%, RAM={ram_usage:.1f}MB")

        self.report()

    def report(self):
        avg_latency = np.mean(self.latencies)
        p95_latency = np.percentile(self.latencies, 95)
        actions_per_sec = 1.0 / avg_latency

        print("\n--- Benchmark Report ---")
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"p95 Latency:     {p95_latency:.4f} s")
        print(f"Actions/Second:  {actions_per_sec:.2f}")
        print("------------------------")

# 使用示例 (伪代码):
# dummy_model = YourVLAModel()
# benchmark = VLABenchmark(dummy_model)
# benchmark.run_benchmark(image, "pick up the red block")
# server.py
from fastapi import FastAPI, UploadFile, File, Form
import uvicorn
from pydantic import BaseModel

app = FastAPI()


# 假设这里加载了大型 VLA 模型
# model = LoadHeavyVLAModel()

@app.post("/predict_action/")
async def predict_action(instruction: str = Form(...), image: UploadFile = File(...)):
    # 1. 读取图像
    image_bytes = await image.read()
    # 2. 图像预处理
    # processed_img = preprocess(image_bytes)

    # 3. 模型推理
    # action_vector = model.generate_action(processed_img, instruction)

    # 模拟 action 向量返回
    action_vector = [0.5, -0.2, 0.1, 0.0, 0.0, 1.0]

    return {"action": action_vector, "status": "success"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# client.py
import requests
import time


def request_remote_action(image_path, instruction, server_url):
    url = f"{server_url}/predict_action/"

    start_time = time.time()

    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        data = {"instruction": instruction}

        # 发送请求到云端/主机
        response = requests.post(url, files=files, data=data)

    end_time = time.time()
    network_latency = end_time - start_time

    if response.status_code == 200:
        result = response.json()
        print(f"Received action: {result['action']}")
        print(f"Total Remote Latency (including network): {network_latency:.4f}s")
        return result['action']
    else:
        print("Error in remote inference")
        return None

# 使用示例:
# SERVER_IP = "http://192.168.1.100:8000"
# action = request_remote_action("camera_capture.jpg", "move forward", SERVER_IP)

import torch
import time


# ... (省略外层类定义，基于之前的 VLABenchmark) ...

def run_benchmark_with_protection(self, sample_image, instruction, num_runs=100):
    print("Starting protected benchmark...")

    for i in range(num_runs):
        start_time = time.time()

        try:
            # 尝试执行模型推理
            action = self.model.predict(sample_image, instruction)

            # 记录成功的时间
            latency = time.time() - start_time
            self.latencies.append(latency)

        except torch.cuda.OutOfMemoryError:
            # 【保护机制】捕获显存溢出错误，防止程序崩溃
            print(f"Run {i}: [警告] 发生 OOM (显存溢出)！正在执行保护性清理...")

            # 清理 PyTorch 的显存缓存，尝试恢复
            torch.cuda.empty_cache()

            # 可以选择记录失败次数，或者暂停几秒钟让系统喘息
            time.sleep(2)
            continue  # 跳过这次，继续下一次循环

        except Exception as e:
            # 捕获其他未知错误（例如图像读取失败、指令格式不对）
            print(f"Run {i}: [错误] 发生未知异常: {e}")
            break  # 如果是严重错误，安全退出循环

    self.report()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader

app = FastAPI()

# 定义您的保护密钥 (实际项目中应写在环境变量里)
SECRET_API_KEY = "jetson-vla-secret-token-2024"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# 验证函数
async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != SECRET_API_KEY:
        # 如果密钥不对，直接拒绝访问，保护算力资源
        raise HTTPException(status_code=403, detail="拒绝访问：无效的 API Key")
    return api_key

# 在接口上强制要求 verify_api_key
@app.post("/predict_action/")
async def predict_action(
    instruction: str = Form(...),
    image: UploadFile = File(...),
    api_key: str = Depends(verify_api_key) # <--- 保护锁在这里
):
    # 只有密钥正确，才会执行到这里的模型推理逻辑
    action_vector = [0.5, -0.2, 0.1, 0.0, 0.0, 1.0]
    return {"action": action_vector, "status": "success"}


import requests


def request_remote_action_protected(image_path, instruction, server_url):
    url = f"{server_url}/predict_action/"

    # 【请求保护】在 Header 中带上我们约定好的密钥
    headers = {
        "X-API-Key": "jetson-vla-secret-token-2024"
    }

    with open(image_path, "rb") as f:
        files = {"image": (image_path, f, "image/jpeg")}
        data = {"instruction": instruction}

        # 将 headers 一起发过去
        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        return response.json()['action']
    else:
        print(f"请求被拒绝或发生错误，状态码: {response.status_code}")
        return None
