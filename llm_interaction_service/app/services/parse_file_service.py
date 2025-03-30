import json
import re
import logging
import fitz
from fastapi import UploadFile, HTTPException
from typing import Dict, Any, List
from langchain_chroma import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import config
from app.factories.ocr_service_factory import OCRServiceFactory
from app.factories.llm_interaction_service_factory import LlmInteractionServiceFactory
from app.factories.pdf_to_image_service_factory import PDFToImageServiceFactory
from app.interfaces.parse_file_service_interface import ParseFileServiceInterface

logger = logging.getLogger(__name__)

class ParseFileService(ParseFileServiceInterface):
    def __init__(self, ollama_base_url: str = "http://llm_host_service:11434", groq_api_key: str = None):
        """
        Initialize the ParseFileService class

        Args:
            ollama_base_url (str): Base URL for Ollama service.
            groq_api_key (str, optional): API key for Groq service.
        """
        self._llm_service = None
        self.pdf_to_image_service = PDFToImageServiceFactory.create_pdf_to_image_service(
            service_name=config.pdf_to_image_service
        )
        self.ocr_service = OCRServiceFactory.create_ocr_service(
            service_name=config.ocr_processing_service
        )
        self.ollama_base_url = ollama_base_url
        self.groq_api_key = groq_api_key

    async def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from a PDF using PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # TODO: DELETE
            logger.info(f"ParseFileService - opened stream of bytes")

            extracted_text = "\n".join([page.get_text("text") for page in doc])
            extracted_text = extracted_text.strip()

            # TODO: DELETE
            logger.info(f"ParseFileService - Length of extracted text: {len(extracted_text)}")

            if len(extracted_text) < 50:
                return "__SCANNED_DOCUMENT__"

            return extracted_text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF")
            raise HTTPException(status_code=500, detail="Error extracting text from PDF.")

    async def process_with_ocr(self, pdf_bytes: bytes) -> str:
        """Process a scanned PDF using OCR."""
        logger.info("Converting PDF to images for OCR processing")
        images = await self.pdf_to_image_service.convert_pdf_to_images(pdf_bytes, enhance=True)

        if not images:
            raise HTTPException(status_code=500, detail="Failed to convert PDF to images")

        logger.info(f"Starting OCR processing on {len(images)} images")
        extracted_text = await self.ocr_service.extract_text_from_multiple_images(images)

        if not extracted_text:
            logger.warning("OCR processing completed but no text was extracted")
            raise HTTPException(
                status_code=400,
                detail="The document appears to be blank or OCR could not extract text"
            )

        logger.info(f"OCR processing completed. Extracted {len(extracted_text)} characters.")
        return extracted_text

    def _create_parse_prompt(self, extracted_text: str) -> Dict[str, str]:
        """Create a prompt for parsing invoice data."""
        context = f"""
            Invoice text data:
            {extracted_text}
        """
        user_input = """
            Extract and structure data from the provided invoice text.
            The returned data should be of type JSON blob.
            The invoice items should also contain any TAXES ITEMS found.
            # The JSON must have the following structure:
            {
                "number": "<invoice_number>",
                "date": "<invoice_date>",
                "dueDate": "<due_date>",
                "total": "<invoice_total>",
                "items": [
                    {
                        "description": "<item_description>",
                        "quantity": "<item_quantity>",
                        "unit_price": "<item_unit_price>",
                    },
                    ...,
                    {
                        "description": "TAX",
                        "quantity": "1",
                        "unit_price": "<tax_amount>",
                    },
                ]
            }
        """

        return {"system": context, "user": user_input}

    def _create_custom_prompt(self, extracted_text: str, user_prompt: str) -> Dict[str, str]:
        """Create a custom prompt based on user input."""
        system = f"""
            System:\n
            You are an assistant that helps users extract specific information from a file.
            The return data should be in html format so that it can be displayed in a web app using "innerHTML".

            File Context:\n{extracted_text}
        """ 
        user = f"""
            User Prompt: {user_prompt}"
        """

        return {"system": system, "user": user}

    async def _process_with_rag(self, extracted_text: str, prompt: str, model: str) -> str:
        """Process file with RAG approach (for Groq service)."""
        try:
            embeddings = SentenceTransformerEmbeddings()
            vectorstore = Chroma(embedding_function=embeddings)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(extracted_text)
            vectorstore.add_texts(chunks)

            retrieved_docs = vectorstore.similarity_search(prompt, k=5)
            relevant_text = "\n".join([doc.page_content for doc in retrieved_docs])

            # Combine the relevant text with the original prompt
            context = f"""
                Invoice text data:
                {relevant_text}
            """
            user_input = f"""
                {prompt}.
                The returned data should be of type JSON blob.
                # The JSON must have the following structure:
                {{
                    "response": "single string with the response to the prompt",
                }}
            """

            result = ""
            async for chunk in self._llm_service.generate_completion(
                    model=model,
                    prompt=({"system": context, "user": user_input}),
                    stream=False
            ):
                result += chunk["response"]

            cleaned_text = re.sub(r"""^```json\n|\n"'```$""""", "", result).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"ParseInvoiceService - Failed to parse JSON response: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing with RAG: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing with RAG: {str(e)}")

    async def process_file(
            self,
            model: str,
            file: UploadFile,
            processing_type: str,
            prompt: str = None,
            ai_service: str = "ollama_local"
    ) -> str:
        """
        Process file using the specified AI service.

        Args:
            model (str): The model to use.
            file (UploadFile): The file to process.
            processing_type (str): Type of processing ("parse" or "prompt").
            prompt (str, optional): Custom prompt for LLM. Required for "prompt" type.
            ai_service (str): AI service to use ("ollama_local" or "groq_cloud").

        Returns:
            Dict[str, Any]: The processed file data.
        """
        self._llm_service = LlmInteractionServiceFactory.create_llm_interaction_service(
            ai_service,
            self.ollama_base_url,
            self.groq_api_key
        )

        # Extract text from PDF
        pdf_bytes = await file.read()
        # TODO: DELETE
        logger.info(f"ParseFileService - No. of PDF Bytes: {len(pdf_bytes)}")

        extracted_text = await self.extract_text_from_pdf(pdf_bytes)
        # TODO: DELETE
        logger.info(f"ParseFileService - extracted text: {extracted_text}")

        if extracted_text == "__SCANNED_DOCUMENT__":
            extracted_text = await self.process_with_ocr(pdf_bytes)

        # Process the invoice based on processing type
        if processing_type == "parse":
            # Create parse prompt
            parse_prompt = self._create_parse_prompt(extracted_text)

            # Generate completion
            result = ""
            async for chunk in self._llm_service.generate_completion(model=model, prompt=parse_prompt, stream=False):
                result += chunk["response"]

            # Clean and parse the result
            cleaned_text = re.sub(r"""^```json\n|\n"'```$""""", "", result).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse JSON response: {str(e)}")

        elif processing_type == "prompt":
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required for 'prompt' type.")

            # For Groq with RAG
            if ai_service == "groq_cloud":
                return await self._process_with_rag(extracted_text, prompt, model)

            # For Ollama or other services
            custom_prompt = self._create_custom_prompt(extracted_text, prompt)

            results = ""
            async for chunk in self._llm_service.generate_completion(model=model, prompt=custom_prompt, stream=False):
                results += chunk

            cleaned_text = re.sub(r"""^```json\n|\n"'```$""""", "", results).strip()

            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=500, detail=f"Failed to parse JSON response: {str(e)}")

        else:
            raise HTTPException(status_code=400, detail="Invalid processing type. Use 'parse' or 'prompt'.")