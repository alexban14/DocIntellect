import logging
from typing import Optional

from app.interfaces.ocr_service_interface import OCRServiceInterface
from app.services.tesseract_ocr_service import TesseractOCRService
# Import other potential OCR services here in the future
# from .azure_ocr_service import AzureOCRService
# from .google_ocr_service import GoogleOCRService

logger = logging.getLogger(__name__)

class OCRServiceFactory:
    """
    Factory class for creating OCR services based on configuration.
    Allows easy addition of new OCR service implementations.
    """

    @staticmethod
    def create_ocr_service(
        service_name: str,
        tesseract_cmd: Optional[str] = None
    ) -> OCRServiceInterface:
        """
        Create an OCR service based on the configuration.

        Args:
            service_name (str): Name of the OCR service to create
            tesseract_cmd (Optional[str]): Path to tesseract executable

        Returns:
            OCRServiceInterface: An instance of the specified OCR service
        """
        service_name = service_name.lower()

        if service_name == 'tesseract':
            logger.info("Creating Tesseract OCR Service")
            return TesseractOCRService(tesseract_cmd)
        # Add other service conditions here in the future
        # elif service_name == 'azure':
        #     return AzureOCRService(config)
        # elif service_name == 'google':
        #     return GoogleOCRService(config)

        raise ValueError(f"Unsupported OCR service: {service_name}")