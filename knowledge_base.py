"""
Restaurant Knowledge Base

This module provides functionality for working with restaurant data.
This is a compatibility layer that imports from the modular knowledge_base package.
"""

# Import from modular structure
from knowledge_base.restaurant import Restaurant
from knowledge_base.base import RestaurantKnowledgeBase
from knowledge_base.scraper_utils import run_scraper_if_needed
from knowledge_base.kb_manager import KnowledgeBaseManager
from knowledge_base import main

# Re-export the key classes and functions
__all__ = ['Restaurant', 'RestaurantKnowledgeBase', 'run_scraper_if_needed', 
           'KnowledgeBaseManager', 'main']

# For backward compatibility
if __name__ == "__main__":
    main()