import logging
import pytesseract
import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
import io
import asyncio
from typing import List, Optional, Dict, Any
from app.interfaces.ocr_service_interface import OCRServiceInterface

logger = logging.getLogger(__name__)

class TesseractOCRService(OCRServiceInterface):
    """Service for extracting text from images using OCR."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the OCR service.

        Args:
            tesseract_cmd (Optional[str]): Path to tesseract executable if not in PATH
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

        # Test if Tesseract is available
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract OCR: {str(e)}")
            logger.warning("Make sure Tesseract OCR is installed and available in PATH")

    async def extract_text_from_image(self, image_bytes: bytes, lang: str = "eng") -> str:
        """
        Extract text from an image using OCR.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Extracted text
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))

            # Extract text with pytesseract
            text = pytesseract.image_to_string(image, lang=lang)

            return text.strip()

        except Exception as e:
            logger.error(f"Failed to extract text from image: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting text with OCR: {str(e)}")

    async def extract_text_from_multiple_images(self, image_bytes_list: List[bytes], lang: str = "eng") -> str:
        """
        Extract text from multiple images and combine the results.

        Args:
            image_bytes_list (List[bytes]): List of images as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Combined extracted text
        """
        if not image_bytes_list:
            return ""

        texts = []

        for i, img_bytes in enumerate(image_bytes_list):
            logger.info(f"Processing image {i+1} of {len(image_bytes_list)}")
            text = await self.extract_text_from_image(img_bytes, lang)
            texts.append(text)

        return "\n\n".join(texts)

    async def process_image_file(self, file: UploadFile, lang: str = "eng") -> str:
        """
        Process an image file uploaded through FastAPI.

        Args:
            file (UploadFile): The uploaded image file
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            str: Extracted text
        """
        try:
            image_bytes = await file.read()

            if not image_bytes:
                raise HTTPException(status_code=400, detail="Empty image file")

            return await self.extract_text_from_image(image_bytes, lang)

        except Exception as e:
            logger.error(f"Failed to process image file: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing image file: {str(e)}")

    async def extract_text_with_confidence(self, image_bytes: bytes, lang: str = "eng") -> Dict[str, Any]:
        """
        Extract text from an image and include confidence scores.

        Args:
            image_bytes (bytes): The image as bytes
            lang (str): Language code for OCR (default: 'eng')

        Returns:
            Dict[str, Any]: Extracted text with confidence data
        """
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get OCR data including confidence
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

            # Process the data to include confidence
            text_blocks = []
            confidence_sum = 0
            confidence_count = 0

            for i in range(len(data['text'])):
                if data['text'][i].strip():
                    text_blocks.append({
                        'text': data['text'][i],
                        'confidence': data['conf'][i],
                        'block_num': data['block_num'][i],
                        'line_num': data['line_num'][i]
                    })

                    if data['conf'][i] > 0:  # Only count valid confidence scores
                        confidence_sum += data['conf'][i]
                        confidence_count += 1

            # Calculate average confidence
            avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0

            # Construct full text
            full_text = ' '.join([block['text'] for block in text_blocks])

            return {
                'text': full_text,
                'avg_confidence': avg_confidence,
                'blocks': text_blocks
            }

        except Exception as e:
            logger.error(f"Failed to extract text with confidence: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting text with confidence: {str(e)}")
