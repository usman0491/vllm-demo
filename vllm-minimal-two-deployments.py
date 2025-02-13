import os
import time
import logging
from typing import Dict
import asyncio

from fastapi import FastAPI, HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

import ray
from ray import serve
from ray.autoscaler.sdk import request_resources

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray.serve")

@ray.remote  # Ensure it runs on a GPU worker node (num_cpus=1, num_gpus=1)
class LLMEngineActor:
    def __init__(self, engine_args: AsyncEngineArgs):
        logger.info("Initializing LLM Engine on a worker node...")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat = None
        logger.info("LLM Engine initialized successfully.")


    async def get_chat_response(self, request_dict: dict, Response_role: str):
        try:
            request = ChatCompletionRequest(**request_dict)
            logger.info(f"Processing request: {request}")

            # Ensure OpenAIServingChat is initialized
            if not self.openai_serving_chat:
                logger.info("Initializing OpenAIServingChat...")
                model_config = await self.engine.get_model_config()
                logger.info(f"Model config retrieved: {model_config}")

                served_model_names = self.engine_args.served_model_name or [self.engine_args.model]
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    served_model_names=served_model_names,
                    response_role = Response_role,
                    lora_modules=[],  # Dummy value for LoRA modules
                    prompt_adapters=None,  # Dummy value for prompt adapters
                    request_logger=None,  # Dummy value for request logger
                    chat_template=None,  # Dummy value for chat template
                )
                logger.info(f"OpenAIServingChat initialized.")
                
            # Call the chat completion function
            logger.info("Calling create_chat_completion()...")
            generator = await self.openai_serving_chat.create_chat_completion(request)
            logger.info("create_chat_completion() executed successfully.")

            # Handle errors
            if isinstance(generator, ErrorResponse):
                logger.warning(f"Error in completion generation: {generator}")
                return {"error": generator.model_dump(), "status_code": generator.code}

            # Ensure correct response format
            assert isinstance(generator, ChatCompletionResponse)
            logger.info(f"Returning JSON response to the client.")
            return generator.model_dump()
        except Exception as e:
            logger.error(f"Exception in get_chat_response: {e}", exc_info=True)
            return {"error": str(e), "status_code": 500}


app = FastAPI()
actor_registry = {}

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.engine_actor = None  # Will hold the remote actor reference

        self.last_request_time = time.time()  # Track last request timestamp
        self.shutdown_timeout = 360  # Set timeout (e.g., 5 minutes)

        # Start the background monitoring task
        self._start_inactivity_monitor()

        # app.add_event_handler("startup", self.startup_event)

    def _start_inactivity_monitor(self):
        """Start a background task to monitor inactivity and shut down the worker node."""
        asyncio.create_task(self._monitor_inactivity())

    async def _monitor_inactivity(self):
        """Periodically check if the worker node has been idle for too long and shut it down."""
        while True:
            await asyncio.sleep(60)  # Check every 60 seconds
            if self.engine_actor:
                idle_time = time.time() - self.last_request_time
                if idle_time > self.shutdown_timeout:
                    logger.info(f"No requests received for {self.shutdown_timeout} seconds. Shutting down worker node.")
                    ray.kill(self.engine_actor)
                    self.engine_actor = None
                    request_resources(bundles=[])
                    logger.info("Worker node shut down successfully.")


    async def _ensure_engine_actor(self):
        global actor_registry
        try:
            self.engine_actor = ray.get_actor("llm_actor")
        except ValueError:        
            request_resources(
                bundles=[{"CPU": 2, "GPU": 1}])
            # time.sleep(60)
            while True:
                resources = ray.available_resources()
                if resources.get("GPU", 0) > 0:  # Check if a worker with GPU exists
                    logger.info("Worker node detected. Initializing engine...")
                    self.engine_actor = LLMEngineActor.options(
                        name="llm_actor", scheduling_strategy="SPREAD", lifetime="detached"
                    ).remote(self.engine_args)
                    actor_registry["llm_actor"] = self.engine_actor
                    self.last_request_time = time.time()  # Reset the timer on each request
                    logger.info("AsyncLLMEngine initialized successfully.")
                    break
                else:
                    logger.info("No worker nodes yet. Waiting...")
                    time.sleep(10)  # Wait before checking again


    # async def startup_event(self):
    #     logger.info("Startup event triggered.")
    #     await self._ensure_engine_actor()


    @app.post("/v1/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        logger.info(f"Ensuring if the engine actor is UP")
        await self._ensure_engine_actor()  # Ensure the engine actor is up
        self.last_request_time = time.time()  # Reset the timer on each request

        logger.info(f"Sending request to LLMEngineActor: {request.dict()}")
        response = await self.engine_actor.get_chat_response.remote(request.dict(), self.response_role)
        # response = await self.engine_actor.test_function.remote()
        logger.info(f"Request to LLMEngineActor completed: {request.dict()}")

        # Handle error response
        if "error" in response:
            logger.warning(f"Error from engine actor: {response['error']}")
            return JSONResponse(content={"error": response["error"]}, status_code=500)
        
        # Return response as JSON
        logger.info("Returning JSON response to the client.")
        return JSONResponse(content=response)




def parse_vllm_args(cli_args: dict[str, str]):
    try:
        logger.info(f"Parsing CLI arguments: {cli_args}")
        parser = FlexibleArgumentParser(description="vLLM CLI")
        parser = make_arg_parser(parser)
        arg_strings = [f"--{key}={value}" for key, value in cli_args.items()]
        parsed_args = parser.parse_args(args=arg_strings)
        logger.info("CLI arguments parsed successfully.")
        return parsed_args
    except Exception as e:
        logger.error(f"Failed to parse CLI arguments: {e}", exc_info=True)
        raise

def build_app(cli_args: Dict[str, str]) -> serve.Application:
    try:
        logger.info(f"Building app with CLI arguments: {cli_args}")
        parsed_args = parse_vllm_args(cli_args)
        engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
        engine_args.worker_use_ray = True
        logger.info("Application built successfully.")
        return VLLMDeployment.bind(engine_args, parsed_args.response_role)
    except Exception as e:
        logger.error(f"Failed to build application: {e}", exc_info=True)
        raise

try:
    model = build_app({
        "model": os.environ.get('MODEL_ID', 'default-model-id'),
        "tensor-parallel-size": os.environ.get('TENSOR_PARALLELISM', '1'),
        "pipeline-parallel-size": os.environ.get('PIPELINE_PARALLELISM', '1'),
    })
    logger.info("Model deployment initialized successfully.")
except Exception as e:
    logger.critical(f"Model deployment failed: {e}", exc_info=True)
    raise
