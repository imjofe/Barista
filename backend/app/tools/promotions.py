"""Mock promotions API tool."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def get_daily_promotion() -> dict[str, dict[str, str]]:
    """
    Retrieve the current daily special/promotion.

    This is a mock function that simulates calling an external Promotions API.
    In production, this would make an HTTP request to a real promotions service.

    Returns:
        Dictionary with 'special' containing 'name' and 'deal' fields.
    """
    # Mock response as specified in the brief
    return {
        "special": {
            "name": "Espresso Elixir",
            "deal": "Get a free pastry with any purchase!",
        }
    }

