import os
import subprocess

from ray import serve
from ray.serve.handle import DeploymentHandle
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response
import httpx  # Use httpx for async HTTP requests

app = FastAPI()

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

@serve.deployment
@serve.ingress(app)  # Initialize FastAPI app and bind with Ray Serve
class VLLMDeployment:
    def __init__(self):
#        self.vllm_service = VLLMConfig.bind()
        self.client = httpx.AsyncClient()  # Initialize HTTP client

    @app.post("/v1/completions")
    async def create_completion(self, request: Request):
        """Forward the request to the vLLM container's endpoint."""
        try:
            # Read the incoming JSON payload
            payload = await request.json()

            # Forward the request to the vLLM container
            response = await self.client.post("http://localhost:8000/v1/completions", json=payload)

            # Return the response from vLLM container directly
            return Response(content=response.content, status_code=response.status_code, media_type=response.headers.get('Content-Type', 'application/json'))

        except httpx.RequestError as exc:
            # Handle connection errors
            return JSONResponse(
                status_code=500,
                content={"error": f"An error occurred while forwarding the request: {exc}"}
            )

# Bind the deployment graph
deployment_graph = VLLMDeployment.bind()