"""Image generation tool using FLUX 1.1 on Azure Foundry."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

import structlog

logger = structlog.get_logger(__name__)


def create_image_gen_tool(
    api_key: str | None,
    endpoint: str,
    deployment_name: str,
    api_version: str = "2025-04-01-preview",
):
    """Create image generation tool with bound Azure OpenAI credentials."""
    @tool
    async def generate_drink_image(drink_name: str) -> dict[str, str | None]:
        """
        Generate an image of a coffee drink using FLUX 1.1 on Azure Foundry.

        Args:
            drink_name: Name of the drink to generate (e.g., "Caramel Delight", "Mocha Magic")

        Returns:
            Dictionary with 'image_url' (if successful) or 'error' message.
        """
        if not api_key:
            return {
                "image_url": None,
                "error": "Image generation API key not configured. Please set AZURE_OPENAI_API_KEY.",
            }

        prompt = f"A beautiful, professional photograph of a {drink_name} coffee drink in a ceramic cup, warm lighting, coffee shop ambiance, high quality, detailed"

        # FLUX 1.1 uses OpenAI-compatible API format
        url = f"{endpoint}/openai/deployments/{deployment_name}/images/generations"
        params = {"api-version": api_version}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    url,
                    params=params,
                    headers={
                        "api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "prompt": prompt,
                        "n": 1,
                        "size": "1024x1024",
                        "response_format": "b64_json",
                    },
                )
                response.raise_for_status()
                data = response.json()

                # FLUX returns base64 images in data array
                if "data" in data and len(data["data"]) > 0:
                    image_base64 = data["data"][0].get("b64_json")
                    if image_base64:
                        # In a real implementation, you'd upload this to S3/CloudFront and return a URL
                        # For MVP, we'll return a data URL (not ideal for production)
                        image_url = f"data:image/png;base64,{image_base64}"
                        logger.info("image_gen.success", drink=drink_name)
                        return {"image_url": image_url, "error": None}
                    else:
                        return {
                            "image_url": None,
                            "error": "Image generation succeeded but no image data was returned.",
                        }
                else:
                    return {
                        "image_url": None,
                        "error": "Image generation API returned unexpected response format.",
                    }

        except httpx.HTTPStatusError as e:
            logger.error("image_gen.http_error", status=e.response.status_code, drink=drink_name)
            error_detail = ""
            try:
                error_data = e.response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except Exception:
                pass
            return {
                "image_url": None,
                "error": f"Image generation API error: {e.response.status_code}. {error_detail}",
            }
        except Exception as e:
            logger.error("image_gen.error", error=str(e), drink=drink_name)
            return {
                "image_url": None,
                "error": f"Image generation failed: {str(e)}",
            }

    return generate_drink_image

