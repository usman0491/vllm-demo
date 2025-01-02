from ray import serve
from ray.serve.handle import DeploymentHandle
import subprocess
import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging

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

    @app.post("/vllm_completions")
    async def get_vllm_completions(self, request: Request):
        # Retrieve the JSON payload from the incoming request
        try:
            payload = await request.json()
        except Exception as e:
            return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

        try:
            # Forward the request to the vLLM container running on localhost
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
        # Debugging: Log incoming request
        try:
            payload = await request.json()
            logging.info(f"Received payload: {payload}")
        except Exception as e:
            logging.error(f"Error parsing JSON: {e}")
            return JSONResponse({"error": "Invalid or missing JSON payload"}, status_code=400)

        # Forward the payload to the vLLM service
        try:
            result = await self.vllm_service.remote(payload)
            return JSONResponse(result)
        except Exception as e:
            logging.error(f"Error in vLLM service call: {e}")
            return JSONResponse({"error": "Internal server error"}, status_code=500)


deployment_graph = VLLMDeployment.bind()
