from fastapi import FastAPI
from app.core.app_logger import setup_logging
from app.core.middleware import setup_cors
import logging
from app.api.endpoints import hello
from app.api.endpoints.interaction import process_file
from app.factories.ocr_service_factory import OCRServiceFactory
from app.core.constants import OCRService


def create_api():
    api = FastAPI()

    @api.on_event("startup")
    async def startup_event():
        # Initialize PaddleOCR service to download the model
        try:
            logging.info("Initializing PaddleOCR service...")
            OCRServiceFactory.create_ocr_service(OCRService.PADDLE)
            logging.info("PaddleOCR service initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize PaddleOCR service: {e}")

    # Apply middleware
    setup_cors(api)

    setup_logging()
    logging.info("logging works!")

    api.include_router(hello.router)

    # llm interaction endpoints
    llmInteractionPrefix = '/llm-interaction-api/v1'
    api.include_router(process_file.router, prefix=llmInteractionPrefix, tags=['LLmInteractionApi', 'LlmProcessFile'])

    return api
