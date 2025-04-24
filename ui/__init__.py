"""
Restaurant Assistant UI Package

This package provides the user interface components for the restaurant recommendation system.
"""

from pathlib import Path

# Define package metadata
UI_DIR = Path(__file__).parent
ASSETS_DIR = UI_DIR / "assets"

# Create assets directory if it doesn't exist
ASSETS_DIR.mkdir(exist_ok=True)

# Export key functions
from .app import (
    initialize_rag,
    display_chat_message,
    process_restaurant_data,
    process_dish_results
)

__all__ = [
    'initialize_rag',
    'display_chat_message',
    'process_restaurant_data',
    'process_dish_results'
]