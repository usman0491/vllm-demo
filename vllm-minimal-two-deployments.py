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


# @ray.remote(num_cpus=2, num_gpus=1)  # Ensure it runs on a GPU worker node 
# class LLMEngineActor:
#     def __init__(self, engine_args: AsyncEngineArgs):
#         logger.info("Initializing LLM Engine on a worker node...")
#         self.engine = AsyncLLMEngine.from_engine_args(engine_args)
#         self.openai_serving_chat = None
#         logger.info("LLM Engine initialized successfully.")


app = FastAPI()
@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str):
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.engine_actor = None  # Will hold the remote actor reference
        # self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        # app.add_event_handler("startup", self.startup_event)
        ray.get(self._ensure_engine_actor())

    async def _ensure_engine_actor(self):
        # self.engine_actor = LLMEngineActor.remote(self.engine_args)
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
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
