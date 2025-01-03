import os
from fastapi import FastAPI
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse
from vllm.utils import FlexibleArgumentParser

import logging
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

    @app.post("/v1/chat/completions")
    async def create_chat_completion(self, request: ChatCompletionRequest):
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            self.openai_serving_chat = OpenAIServingChat(
                self.engine,
                model_config,
                response_role=self.response_role,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(request)
        return ChatCompletionResponse(model_dump=generator.model_dump())

def parse_vllm_args(cli_args: dict):
    parser = FlexibleArgumentParser(description="vLLM CLI")
    parsed_args = parser.parse_args(args=[f"--{k}={v}" for k, v in cli_args.items()])
    return parsed_args

def build_app(cli_args: dict) -> serve.Application:
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    return VLLMDeployment.bind(engine_args, parsed_args.response_role)

model = build_app({
    "model": os.environ['MODEL_ID'],
    "tensor-parallel-size": os.environ['TENSOR_PARALLELISM'],
    "pipeline-parallel-size": os.environ['PIPELINE_PARALLELISM']
})
