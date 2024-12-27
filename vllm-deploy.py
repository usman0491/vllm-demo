from ray import serve
from ray.serve.handle import DeploymentHandle
import subprocess
import os

# These imports are used only for type hints:
from typing import Dict
import requests
from starlette.requests import Request
from starlette.responses import JSONResponse

@serve.deployment
class VLLMDeployment:
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
        #self.base_url = "http://localhost:8000"
        
    def __del__(self):
        subprocess.run(["docker", "stop", self.container_id], check=True)
        print(f"VLLM container stopped: {self.container_id}")

    async def __call__(self, request: Request):
        sub_path = request.url.path[len("/vllm"):]
        target_url = f"http://localhost:8000/vllm{sub_path}"

        try:
            response = requests.request(
                method=request.method,
                url=target_url,
                headers=request.headers,
                data=await request.body(),
                timeout=10,
            )

            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=dict(response.headers),
            )

        except requests.exceptions.RequestException as e:
            return JSONResponse({"error": str(e)}, status_code=500)
deployment_graph = VLLMDeployment.bind()