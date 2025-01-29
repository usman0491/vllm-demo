import os
import time
import logging
from typing import Dict, Optional, List

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

app = FastAPI()

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str):
        try:
            logger.info(f"Initializing VLLMDeployment with engine args: {engine_args}")
            self.openai_serving_chat = None
            self.engine_args = engine_args
            self.response_role = response_role
            # self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            self.engine = None
            logger.info(f"VLLMDeployment engine initialized successfully (engine is not yet created).")
        except Exception as e:
            logger.error(f"Failed to initialize VLLMDeployment: {e}", exc_info=True)
            raise

    async def _initialize_engine(self):
        """Lazy initialization of the AsyncLLMEngine."""
        if not self.engine:
            try:
                logger.info("Checking for available worker nodes...")
                while True:
                    resources = ray.available_resources()
                    if resources.get("nvidia.com/gpu", 0) > 0:  # Check if a worker with GPU exists
                        logger.info("Worker node detected. Initializing engine...")
                        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
                        logger.info("AsyncLLMEngine initialized successfully.")
                        break
                    else:
                        logger.info("No worker nodes yet. Waiting...")
                        time.sleep(10)  # Wait before checking again
            except Exception as e:
                logger.error(f"Failed to initialize AsyncLLMEngine: {e}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Error initializing engine: {str(e)}"
                )

    @app.post("/v1/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest, raw_request: Request):
        try:
            logger.info(f"Received completion request: {request}")

            resources = ray.available_resources()
            if resources.get("GPU", 0) == 0:
                logger.info("No worker detected. Requesting worker node...")
                request_resources(
                    bundles=[{"CPU": 2}, {"GPU": 1}])

            #Ensure the engine is initialized
            await self._initialize_engine()
            
            if not self.openai_serving_chat:
                model_config = await self.engine.get_model_config()
                if self.engine_args.served_model_name is not None:
                    served_model_names = self.engine_args.served_model_name
                else:
                    served_model_names = [self.engine_args.model]
                self.openai_serving_chat = OpenAIServingChat(
                    self.engine,
                    model_config,
                    served_model_names=served_model_names,
                    response_role=self.response_role,
                    lora_modules=[],  # Dummy value for LoRA modules
                    prompt_adapters=None,  # Dummy value for prompt adapters
                    request_logger=None,  # Dummy value for request logger
                    chat_template=None,  # Dummy value for chat template
                )
                logger.info(f"OpenAIServingChat instance initialized.")
            
            generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

            if isinstance(generator, ErrorResponse):
                logger.warning(f"Error in completion generation: {generator}")
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.code
                )
            
            if request.stream:
                logger.info(f"Streaming response back to the client.")
                return StreamingResponse(generator, media_type="text/event-stream")
            else:
                assert isinstance(generator, ChatCompletionResponse)
                logger.info(f"Returning JSON response to the client.")
                return JSONResponse(content=generator.model_dump())
        except HTTPException as http_exc:
            logger.error(f"HTTP exception encountered: {http_exc.detail}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in create_chat_completion: {e}", exc_info=True)
            return JSONResponse(
                content={"error": "Internal server error", "details": str(e)}, 
                status_code=500,
                )

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