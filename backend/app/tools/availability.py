"""Availability checker tool for time-based drink availability."""

from __future__ import annotations

from typing import Literal

import pendulum
from langchain_core.tools import tool

# Availability rules from the brief
AVAILABILITY_RULES: dict[str, tuple[int, int]] = {
    "Mocha Magic": (8, 12),  # 8 AM to 12 PM
    "Vanilla Dream": (12, 16),  # 12 PM to 4 PM
    "Caramel Delight": (16, 20),  # 4 PM to 8 PM
    "Hazelnut Harmony": (8, 14),  # 8 AM to 2 PM
}


@tool
def check_drink_availability(drink_name: str, current_time: str | None = None) -> dict[str, str | bool]:
    """
    Check if a specific drink is currently available based on time-based rules.

    Args:
        drink_name: Name of the drink to check (e.g., "Mocha Magic", "Vanilla Dream")
        current_time: Optional ISO format time string. If not provided, uses current time.

    Returns:
        Dictionary with 'available', 'drink_name', 'current_time', and optional 'message'.
    """
    if current_time:
        now = pendulum.parse(current_time)
    else:
        now = pendulum.now()

    current_hour = now.hour

    # Check if drink has time restrictions
    if drink_name in AVAILABILITY_RULES:
        start_hour, end_hour = AVAILABILITY_RULES[drink_name]
        is_available = start_hour <= current_hour < end_hour

        if is_available:
            message = f"Yes, {drink_name} is available now (available from {start_hour}:00 to {end_hour}:00)."
        else:
            # Suggest alternative
            available_now = [
                name
                for name, (s, e) in AVAILABILITY_RULES.items()
                if s <= current_hour < e
            ]
            if available_now:
                suggestion = f"Perhaps you'd like to try our {available_now[0]}, which is available now?"
            else:
                suggestion = "All time-restricted drinks are currently unavailable. Our classic brews (Espresso Elixir, Latte Lux, Cappuccino Charm) are available all day!"

            message = (
                f"I'm sorry, {drink_name} is only available from {start_hour}:00 to {end_hour}:00. "
                f"{suggestion}"
            )

        return {
            "available": is_available,
            "drink_name": drink_name,
            "current_time": now.to_iso8601_string(),
            "message": message,
        }
    else:
        # All other drinks are available all day
        return {
            "available": True,
            "drink_name": drink_name,
            "current_time": now.to_iso8601_string(),
            "message": f"Yes, {drink_name} is available all day!",
        }

