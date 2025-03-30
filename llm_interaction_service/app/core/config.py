from pydantic_settings import BaseSettings

class Config(BaseSettings):
    service_name: str = 'service_name'
    secret_key: str = 's3cr3t_k3y'
    ocr_processing_service: str = 'tesseract'
    pdf_to_image_service: str = 'pymupdf_opencv_pillow'
    ollama_base_url: str = "http://llm_host_service:11434"
    groq_api_key: str = "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"

config = Config()
