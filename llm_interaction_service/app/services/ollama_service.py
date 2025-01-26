import httpx
from typing import Optional, Dict, Any, AsyncGenerator

class OllamaService:
    def __init__(self, base_url: str, timeout: int = 30):
        """
        Initialize the OllamaService with the base URL and default timeout.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def generate_completion(
        self,
        model: str,
        prompt: str,
        suffix: Optional[str] = None,
        images: Optional[list[str]] = None,
        format: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        template: Optional[str] = None,
        stream: bool = True,
        raw: Optional[bool] = None,
        keep_alive: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Makes a request to the `/api/generate` endpoint for text generation.

        Args:
            model (str): Model name.
            prompt (str): The input prompt.
            suffix (Optional[str]): Suffix to append to the response.
            images (Optional[list[str]]): List of base64-encoded images.
            format (Optional[str]): Response format (`json` or schema).
            options (Optional[Dict[str, Any]]): Advanced model options.
            system (Optional[str]): System message override.
            template (Optional[str]): Prompt template override.
            stream (bool): If True, stream responses.
            raw (Optional[bool]): Use raw prompt format.
            keep_alive (Optional[str]): Time to keep the model in memory.

        Yields:
            dict: JSON object from the API response.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "suffix": suffix,
            "images": images,
            "format": format,
            "options": options,
            "system": system,
            "template": template,
            "stream": stream,
            "raw": raw,
            "keep_alive": keep_alive,
        }

        # Remove None values from the payload
        payload = {key: value for key, value in payload.items() if value is not None}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                
                # Stream response handling
                if stream:
                    async for line in response.aiter_lines():
                        yield httpx.Response.json(line)
                else:
                    yield response.json()

            except httpx.HTTPStatusError as e:
                raise RuntimeError(f"HTTP error occurred: {e.response.text}") from e
            except httpx.RequestError as e:
                raise RuntimeError(f"Request error occurred: {e}") from e
