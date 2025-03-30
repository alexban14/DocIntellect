import logging
from typing import Optional

from app.interfaces.pdf_to_image_service_interface import PDFToImageServiceInterface
from app.services.pymupdf_opencv_pil_pdf_to_image_service import PyMuPDFOpenCvPilPDFToImageService
# Import other potential PDF to image services here in the future
# from .azure_pdf_to_image_service import AzurePDFToImageService
# from .google_pdf_to_image_service import GooglePDFToImageService

logger = logging.getLogger(__name__)

class PDFToImageServiceFactory:
    """
    Factory class for creating PDF to Image services based on configuration.
    Allows easy addition of new PDF to Image service implementations.
    """

    @staticmethod
    def create_pdf_to_image_service(
        service_name: str
    ) -> PDFToImageServiceInterface:
        """
        Create a PDF to Image service based on the configuration.

        Args:
            service_name (str): Name of the PDF to Image service to create

        Returns:
            PDFToImageServiceInterface: An instance of the specified PDF to Image service
        """
        service_name = service_name.lower()

        if service_name == 'pymupdf_opencv_pillow':
            logger.info("Creating PyMuPDF PDF to Image Service")
            return PyMuPDFOpenCvPilPDFToImageService()
        # Add other service conditions here in the future

        raise ValueError(f"Unsupported PDF to Image service: {service_name}")