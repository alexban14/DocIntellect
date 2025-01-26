from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.api.endpoints.interaction.request_schemas.generate_request import GenerateRequest
import logging
from app.services.ollama_service import OllamaService

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency to provide OllamaService
def get_ollama_service() -> OllamaService:
    return OllamaService(base_url="http://localhost:8220", timeout=6000)

@router.post("/generate")
async def generate_text(
    generate_request: GenerateRequest,
    ollama_service: OllamaService = Depends(get_ollama_service)
):
    try:
        response_generator = ollama_service.generate_completion(
            model=generate_request.model,
            prompt=generate_request.prompt,
            stream=generate_request.stream
        )
        if generate_request.stream:
            return StreamingResponse(response_generator, media_type="application/json")
        else:
            results = [chunk async for chunk in response_generator]
            return {"results": results}
    except RuntimeError as e:
        logging.error(f"Error occured during {__name__}")
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))
