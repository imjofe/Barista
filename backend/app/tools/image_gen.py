"""Image generation tool using Stability AI."""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

import structlog

logger = structlog.get_logger(__name__)


def create_image_gen_tool(api_key: str | None):
    """Create image generation tool with bound API key."""
    @tool
    async def generate_drink_image(
        drink_name: str,
        base_url: str = "https://api.stability.ai",
    ) -> dict[str, str | None]:
        """
        Generate an image of a coffee drink using Stability AI.

        Args:
            drink_name: Name of the drink to generate (e.g., "Caramel Delight", "Mocha Magic")
            base_url: Base URL for Stability AI API.

        Returns:
            Dictionary with 'image_url' (if successful) or 'error' message.
        """
        if not api_key:
            return {
                "image_url": None,
                "error": "Image generation API key not configured. Please set STABILITY_API_KEY.",
            }

        prompt = f"A beautiful, professional photograph of a {drink_name} coffee drink in a ceramic cup, warm lighting, coffee shop ambiance, high quality, detailed"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "application/json",
                    },
                    json={
                        "text_prompts": [{"text": prompt}],
                        "cfg_scale": 7,
                        "height": 1024,
                        "width": 1024,
                        "samples": 1,
                        "steps": 30,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # Stability AI returns base64 images in artifacts
                if "artifacts" in data and len(data["artifacts"]) > 0:
                    image_base64 = data["artifacts"][0].get("base64")
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
            return {
                "image_url": None,
                "error": f"Image generation API error: {e.response.status_code}",
            }
        except Exception as e:
            logger.error("image_gen.error", error=str(e), drink=drink_name)
            return {
                "image_url": None,
                "error": f"Image generation failed: {str(e)}",
            }
    
    return generate_drink_image

