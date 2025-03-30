import os
import logging
import json
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from app.core.config import config
from app.services.parse_file_service import ParseFileService

router = APIRouter()
logger = logging.getLogger(__name__)

# Dependency to provide LlmInteractionService
def get_llm_interaction_service() -> ParseFileService:
    ollama_base_url = config.ollama_base_url
    groq_api_key = config.groq_api_key
    return ParseFileService(ollama_base_url=ollama_base_url, groq_api_key=groq_api_key)

@router.post("/process-file")
async def process_invoice(
        model: str = Form(...),
        file: UploadFile = File(...),
        processing_type: str = Form(...),
        prompt: str = Form(None),
        ai_service: str = Form("ollama_local"),
        parse_file_service: ParseFileService = Depends(get_llm_interaction_service)
):
    """
    Process invoice PDF with the specified AI service.

    - "ai_service": Choose between "ollama_local" or "groq_cloud"
    - "processing_type":
        - "parse": Extracts structured invoice details.
        - "prompt": Sends extracted file text to the LLM for a custom response.
    """
    try:
        if ai_service not in ["ollama_local", "groq_cloud"]:
            raise HTTPException(status_code=400, detail="Invalid AI service. Use 'ollama_local' or 'groq_cloud'.")

        result = await parse_file_service.process_file(
            model=model,
            file=file,
            processing_type=processing_type,
            prompt=prompt,
            ai_service=ai_service
        )

        return JSONResponse(content=result)

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"File processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
