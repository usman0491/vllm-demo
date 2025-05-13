import os
import time
import logging
from typing import Dict, List, Optional
import asyncio

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

import ray
from ray import serve
from ray.autoscaler.sdk import request_resources

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

import docker

# from vllm.logger import RequestLogger

# from vllm.entrypoints.openai.serving_engine import LoRAModulePath

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray.serve")


@ray.remote(num_gpus=1)  # Ensure it runs on a GPU worker node (num_cpus=1, num_gpus=1)
class LLMEngineActor:
    def __init__(self, engine_args: dict):
        self.engine_args = engine_args
        self.container = None
        self.container_port = 8000
        self.host_port = self._allocate_port()
        self.base_url = f"http://localhost:{self.host_port}"
        self.docker_client = docker.from_env()
        self._start_container()
    
    def _allocate_port(self):
        import socket
        s = socket.socket()
        s.bind(('', 0))
        port  = s.getsockname()[1]
        s.close()
        return port

    def _start_container(self):
        logger.info(f"Starting vLLM container for model: {self.engine_args.model_name}")
        self.container = self.docker_client.containers.run(
            "vllm/vllm-openai:latest",
            detach=True,
            ports={f'{self.container_port}/tcp': self.host_port},
            environment={
                "HUGGING_FACE_HUB_TOKEN": os.getenv("HUGGING_FACE_HUB_TOKEN", "")
            },
            command=[
                "--model", self.engine_args.model_name,
                "--enforce-eager",
                "--max-model-len", "8000",
                "--max-num-seqs", "10"
            ],
            name=f"vllm-{self.engine_args.model_name.replace('/', '-')}",
            remove=True,
        )
        


    async def get_chat_response(self, request_dict: dict):
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=request_dict,
                    headers={"Authorization": f"Bearer {os.getenv('VLLM_API_KEY', '')}"}
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"Error from vLLM container: {error}")
                        return {"error": error, "status_code": response.status}
                    return await response.json()
        except Exception as e:
            logger.exception("Failed to get chat response")
            return {"error": str(e), "status_code": 500}
        
    def shutdown(self):
        logger.info("Shutting down container for model {self.engine_args.model_name}")
        if self.container:
            self.container.stop()
            self.container = None


app = FastAPI()
@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)

class VLLMDeployment:
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        max_num_seqs: int,
        max_model_len: int,
    ):
        self.engine_actors = {}  # Dictionary to hold the reference/track to the model-specific actor
        self.num_models = 0  # Track the number of active models
        self.active_models = set()  # Track the active models names
        self.last_request_time = {}  # Track last request time per model
        self.shutdown_timeout = 600  # Set timeout (e.g., 30 minutes)
        self.engine_args = dict([
            ("model", model),
            ("tensor_parallel_size", tensor_parallel_size),
            ("pipeline_parallel_size", pipeline_parallel_size),
            ("max_num_seqs", 12),
            ("max_model_len", 8000),
            ("enforce_eager", True),
            ("disable_log_requests", True),
            ("dtype", "auto"),
            ("trust_remote_code", True),
            # ("limit_mm_per_prompt", {"image": 2}),
            # ("mm_processor_kwargs", {"image_size": 224}),
            # ("disable_mm_preprocessor_cache", True),
        ])

        # Start the background monitoring task
        self._start_inactivity_monitor()

    def _start_inactivity_monitor(self):
        """Start a background task to monitor inactivity and shut down the worker node."""
        asyncio.create_task(self._monitor_inactivity())

    async def _monitor_inactivity(self):
        """Periodically check if the worker node has been idle for too long and shut it down."""
        while True:
            await asyncio.sleep(60)  # Check every 60 seconds
            for model_name in list(self.engine_actors.keys()):
                idle_time = time.time() - self.last_request_time.get(model_name, 0)
                if idle_time > self.shutdown_timeout:
                    logger.info(f"No requests for {model_name} in {self.shutdown_timeout} seconds. Shutting down worker node for {model_name}.")
                    self.engine_actors[model_name].shutdown.remote()
                    ray.kill(self.engine_actors[model_name])
                    del self.engine_actors[model_name]
                    self.active_models.discard(model_name)
                    self.num_models -= 1
                    self._update_resource_request()
                    logger.info(f"Worker node for {model_name} shut down successfully.")

    def _update_resource_request(self):
        request_resources(bundles=[{"CPU": 2, "GPU": 1}] * self.num_models)

    async def _ensure_engine_actor(self, model_name: str):
        if model_name in self.engine_actors:
            return
        
        # Set last request time to the current time to avoid immediate shutdown
        self.last_request_time[model_name] = time.time() + 300  # Set to 5 minutes in the future to add more time for initialization
        
        self.active_models.add(model_name)
        self.num_models += 1
        self.engine_args.model_name = model_name # Update model name in engine_args
        self._update_resource_request()
        
        # Set a placeholder to indicate that the model is being initialized
        self.engine_actors[model_name] = None
        asyncio.create_task(self._monitor_resources_and_initialize(model_name))


    async def _monitor_resources_and_initialize(self, model_name: str):
        logger.info(f"Waiting for worker node to become available for model {model_name}...")
        while True:
            resources = ray.cluster_resources()
            if resources.get("GPU", 0) >= self.num_models:
                logger.info(f"Worker node detected for model {model_name}. Initializing engine...")
                self.engine_actors[model_name] = LLMEngineActor.options(
                    name=f"llm_actor_{model_name}", scheduling_strategy="SPREAD", lifetime="detached"
                ).remote(self.engine_args)
                logger.info(f"AsyncLLMEngine for {model_name} initialized successfully.")
                break
            else:
                logger.info(f"No worker node for {model_name} yet, number of models = {self.num_models}, resources = {resources}")
                await asyncio.sleep(10)

    allowed_models = {
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "meta-llama/Llama-Guard-3-11B-Vision",
    }

    @app.post("/v1/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        model_name = request.model # Extract model name from the request

        # Validate model name against the allowed list
        if model_name not in self.allowed_models:
            return JSONResponse(
                content={
                    "error": "Invalid model selection. Please choose from the allowed models.",
                    "Allowed Models": list(self.allowed_models)
                },
                status_code=400
            )

        logger.info(f"Ensuring if the engine actor is UP")
        await self._ensure_engine_actor(model_name)  # Ensure the engine actor is up

        if self.engine_actors[model_name] is None: # add one condition on llm_actor{model_name}' status
            return JSONResponse(
            content={"message": f"Model {model_name} is starting, please try again later."},
            status_code=503
        )

        self.last_request_time[model_name] = time.time() # Reset the timer on each request

        logger.info(f"Sending request to LLMEngineActor: {request.dict()}")
        response = await self.engine_actors[model_name].get_chat_response.remote(request.dict())
        logger.info(f"Request to LLMEngineActor completed: {request.dict()}")

        # Handle error response
        if "error" in response:
            logger.warning(f"Error from engine actor: {response['error']}")
            return JSONResponse(content={"error": response["error"]}, status_code=500)
        
        # Return response as JSON
        logger.info("Returning JSON response to the client.")
        return JSONResponse(content=response)


    @app.get("/models")
    async def list_models(self):
        return JSONResponse(content={
            "num_models": self.num_models,
            "active_models": list(self.active_models),
            "allowed_models": list(self.allowed_models)
        })

deployment = VLLMDeployment.bind(
    model=os.environ.get('MODEL_ID', 'default-model-id'),
    tensor_parallel_size=int(os.environ.get('TENSOR_PARALLELISM', '1')),
    pipeline_parallel_size=int(os.environ.get('PIPELINE_PARALLELISM', '1')),
    max_num_seqs=int(os.environ.get('MAX_NUM_SEQS', '10')),
    max_model_len=int(os.environ.get('MAX_MODEL_LEN', '64000')),
)