import os
import time
import logging
from typing import Dict

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

@ray.remote(num_cpus=2, num_gpus=1)  # Ensure it runs on a GPU worker node 
class LLMEngineActor:
    def __init__(self, engine_args: AsyncEngineArgs):
        logger.info("Initializing LLM Engine on a worker node...")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self.openai_serving_chat = None
        logger.info("LLM Engine initialized successfully.")

    async def get_chat_response(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            logger.info(f"Processing request: {request}")
            # if not self.openai_serving_chat:
            #     model_config = await self.engine.get_model_config()
            #     if self.engine_args.served_model_name is not None:
            #         served_model_names = self.engine_args.served_model_name
            #     else:
            #         served_model_names = [self.engine_args.model]
            #     self.openai_serving_chat = OpenAIServingChat(
            #         self.engine,
            #         model_config,
            #         served_model_names=served_model_names,
            #         response_role=self.response_role,
            #         lora_modules=[],  # Dummy value for LoRA modules
            #         prompt_adapters=None,  # Dummy value for prompt adapters
            #         request_logger=None,  # Dummy value for request logger
            #         chat_template=None,  # Dummy value for chat template
            #     )
            #     logger.info(f"OpenAIServingChat instance initialized.")
                
            # generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

            # if isinstance(generator, ErrorResponse):
            #     logger.warning(f"Error in completion generation: {generator}")
            #     return JSONResponse(
            #         content=generator.model_dump(), status_code=generator.code
            #     )
            
            # if request.stream:
            #     logger.info(f"Streaming response back to the client.")
            #     return StreamingResponse(generator, media_type="text/event-stream")
            # else:
            #     assert isinstance(generator, ChatCompletionResponse)
            #     logger.info(f"Returning JSON response to the client.")
            #     return JSONResponse(content=generator.model_dump())
        except Exception as e:
            logger.error(f"Exception in get_chat_response: {e}", exc_info=True)
            # return JSONResponse(content={"error": "Internal Server Error"}, status_code=500)


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

        app.add_event_handler("startup", self.startup_event)


    async def _ensure_engine_actor(self):
        global actor_registry
        try:
            actor_registry["llm_actor"] = ray.get_actor("llm_actor")
        except ValueError:        
            request_resources(
                bundles=[{"CPU": 2, "GPU": 1}])
            
            time.sleep(120)

            actor_registry["llm_actor"] = LLMEngineActor.options(name="llm_actor").remote(self.engine_args)
        # """Ensures that the LLMEngineActor is running on a worker node."""
        # if self.engine_actor is None:
        #     logger.info("Requesting worker node with GPU...")
        #     request_resources(
        #             bundles=[{"CPU": 2, "GPU": 1}])
        #     while True:
        #         resources = ray.available_resources()
        #         if resources.get("GPU", 0) > 0:
        #             logger.info("Worker node detected. Initializing engine actor...")
        #             self.engine_actor = LLMEngineActor.remote(self.engine_args)
        #             logger.info("LLM Engine Actor initialized.")
        #             break
        #         else:
        #             logger.info("Waiting for worker node with GPU...")
        #             time.sleep(5)

    async def startup_event(self):
        logger.info("Startup event triggered.")
        await self._ensure_engine_actor()


    @app.post("/v1/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        # await self._ensure_engine_actor()  # Ensure the engine actor is up
        request_data = request.json()
        # response = await self.engine_actor.get_chat_response.remote(request_data, raw_request)
        # if "error" in response:
        #     return JSONResponse(content=response["error"], status_code=response["status_code"])
        # return JSONResponse(content=response["response"])




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
