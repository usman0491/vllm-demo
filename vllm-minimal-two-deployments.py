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
# from vllm.entrypoints.openai.serving_engine import LoRAModulePath
from vllm.utils import FlexibleArgumentParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray.serve")


@ray.remote(num_gpus=1)  # Ensure it runs on a GPU worker node (num_cpus=1, num_gpus=1)
class LLMEngineActor:
    def __init__(self, engine_args: AsyncEngineArgs):
        logger.info("Initializing LLM Engine on a worker node...")
        self.engine_args = engine_args
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat = None
        logger.info("LLM Engine initialized successfully.")


    async def get_chat_response(self, request_dict: dict, Response_role: str):
        try:
            # Remove unwanted parameters
            request_dict.pop("logprobs", None)
            request_dict.pop("top_logprobs", None)

            request = ChatCompletionRequest(**request_dict)
            logger.info(f"Processing request: {request}")

            # Ensure OpenAIServingChat is initialized
            if not self.openai_serving_chat:
                logger.info("Initializing OpenAIServingChat...")
                model_config = await self.engine.get_model_config()
                logger.info(f"Model config retrieved: {model_config}")

                #served_model_names = self.engine_args.served_model_name or [self.engine_args.model]
                # models = await self.engine.get_models()
                class DummyModel:
                    def __init__(self, name):
                        self.name = name
                        self.is_base_model = True

                models = DummyModel(self.engine_args.model)

                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    #served_model_names=served_model_names,
                    models,
                    response_role = Response_role,
                    #lora_modules=[],  # Dummy value for LoRA modules
                    # prompt_adapters=None,  # Dummy value for prompt adapters
                    request_logger=None,  # Dummy value for request logger
                    chat_template=None,  # Dummy value for chat template
                    chat_template_content_format=None,
                    # return_tokens_as_token_ids: bool = False,
                    # enable_reasoning: bool = False,
                    # reasoning_parser: Optional[str] = None,
                    # enable_auto_tools: bool = False,
                    # tool_parser: Optional[str] = None,
                    # enable_prompt_tokens_details: bool = False,
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
@serve.deployment(name="VLLMDeployment")
# app.router.redirect_slashes = False
# @serve.deployment(name="VLLMDeployment", route_prefix="/", health_check_timeout_s=300)

@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.engine_actors = {}  # Dictionary to hold the reference/track to the model-specific actor
        self.num_models = 0  # Track the number of active models
        self.active_models = set()  # Track the active models names

        self.last_request_time = {}  # Track last request time per model
        self.shutdown_timeout = 600  # Set timeout (e.g., 30 minutes)

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
        self.engine_args.model = model_name # Update model name in engine_args
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
        "meta-llama/Llama-3.2-11B-Vision-Instruct"
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
        response = await self.engine_actors[model_name].get_chat_response.remote(request.dict(), self.response_role)
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
