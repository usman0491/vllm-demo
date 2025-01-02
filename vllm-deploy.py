from ray import serve
from ray.serve.handle import DeploymentHandle
import subprocess
import os
import requests
from typing import Dict
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment
class VLLMConfig:
    def __init__(self):
        self.container_id = subprocess.check_output(
            [
                "docker", "run", "-d", "--rm",
                "-p", "8000:8000",
                "-e", "HUGGING_FACE_HUB_TOKEN=" + os.getenv("HUGGING_FACE_HUB_TOKEN"),
                "-e", "VLLM_API_KEY=" + os.getenv("VLLM_API_KEY"),
                "vllm/vllm-openai:latest",  # vLLM container image
                "--model", "meta-llama/Llama-3.1-8B-Instruct",
                "--gpu-memory-utilization", "0.95",
                "--enforce-eager"
            ]
        ).decode("utf-8").strip()
        print(f"VLLM container started with ID: {self.container_id}")

    def __del__(self):
        subprocess.run(["docker", "stop", self.container_id], check=True)
        print(f"VLLM container stopped: {self.container_id}")

    async def __call__(self, request: Request):
	    payload = await request.json()
        try:
            response = requests.post(
                "http://localhost:8000/v1/completions",
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY')}"}
            )
            return JSONResponse(response.json())
        except requests.exceptions.RequestException as e:
            return JSONResponse({"error": str(e)}, status_code=500)

@serve.deployment
class VLLMDeployment:
    def __init__(self):
        self.vllm_service = VLLMConfig.bind()

    async def __call__(self, request: Request):
        payload = await request.json()
        return await self.vllm_service.remote(payload)


deployment_graph = VLLMDeployment.bind()
