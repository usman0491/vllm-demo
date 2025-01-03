import os
import subprocess
from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import Request
from starlette.responses import JSONResponse, Response
import httpx

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
        

@serve.deployment
@serve.ingress(app := FastAPI())
class VLLMDeployment:
    def __init__(self):
        self.vllm_service = VLLMConfig.bind()
        self.client = httpx.AsyncClient()

    @app.post("/v1/completions")
    async def create_completion(self, request: Request):
        try:
            payload = await request.json()

            response = await self.client.post(
                "http://localhost:8000/v1/completions",
                json=payload,
                headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY')}"}
            )

            return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get('Content-Type', 'application/json'))
        except httpx.RequestError as exc:
            return JSONResponse(
                status_code=500,
                content={"error": f"An error occurred while forwarding the request: {exc}"}
            )

    async def __call__(self, request: Request):
        return JSONResponse({"message": "vLLM Deployment is running."})


deployment_graph = VLLMDeployment.bind()
