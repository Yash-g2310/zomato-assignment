"""
Restaurant RAG (Retrieval Augmented Generation) System

This module provides functionality for answering natural language questions about restaurants.
This is a compatibility layer that imports from the modular RAG package.
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import from the modular package
from rag import RestaurantRAG, find_latest_json_file, demo_rag

# Re-export for backwards compatibility
__all__ = ['RestaurantRAG', 'find_latest_json_file', 'demo_rag']

# For backwards compatibility
if __name__ == "__main__":
    demo_rag()