from fastapi import FastAPI

from app.core.app_logger import setup_logging
import logging
from app.api.endpoints import hello
from app.api.endpoints.interaction import process_file


def create_api():
    api = FastAPI()

    setup_logging()
    logging.info("logging works!")

    api.include_router(hello.router)

    # llm interaction endpoints
    llmInteractionPrefix = '/llm-interaction-api/v1'
    api.include_router(process_file.router, prefix=llmInteractionPrefix, tags=['LLmInteractionApi', 'LlmProcessFile'])

    return api
