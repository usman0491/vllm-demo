import os

from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

from ray import serve

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

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(name="VLLMDeployment")
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self, engine_args: AsyncEngineArgs, response_role: str):
        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @app.post("/v1/completions")
    async def create_completion(self, request: ChatCompletionRequest, raw_request: Request):
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
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_completion(request, raw_request)
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())

def parse_vllm_args(cli_args: dict[str, str]):
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parser = make_arg_parser(parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.append(f"--{key}={value}")
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args

def build_app(cli_args: dict[str, str]) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True
    return VLLMDeployment.bind(engine_args, parsed_args.response_role)

model = build_app({
    "model": os.environ['MODEL_ID'],
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM']
})
