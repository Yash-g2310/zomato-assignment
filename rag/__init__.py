"""
Restaurant RAG (Retrieval Augmented Generation) Package

This package provides components for creating a RAG system for restaurant data,
enabling natural language queries about restaurants and their menus.
"""

import os
import logging
from typing import Dict, List, Union, Any

from .embeddings import RestaurantEmbeddings
from .document_store import RestaurantDocumentStore
from .query_engine import RestaurantQueryEngine

logger = logging.getLogger(__name__)

def find_latest_json_file():
    """Find the latest restaurant JSON file in the current directory or parent."""
    from config import get_latest_data_file
    import json
    import os
    
    # First try to find combined data files
    latest = get_latest_data_file(prefix="restaurants_")
    if latest:
        # Debug info
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Found JSON file with {len(data)} restaurants: {latest}")
            
            # Add debugging to check the content structure
            if len(data) == 0:
                logger.warning("The JSON file exists but contains an empty array!")
            elif len(data) > 0:
                logger.info(f"Sample restaurant: {list(data[0].keys())}")
                
        except Exception as e:
            logger.error(f"Error examining JSON file {latest}: {e}")
        
        return latest
        
    # Fall back to Zomato-only files
    latest = get_latest_data_file(prefix="zomato_")
    if latest:
        logger.info(f"Using Zomato-only data file: {latest}")
        return latest
        
    # Fall back to Swiggy-only files
    latest = get_latest_data_file(prefix="swiggy_")
    if latest:
        logger.info(f"Using Swiggy-only data file: {latest}")
        return latest
    
    # If no file found, try to create a test file
    logger.warning("No restaurant data files found! Creating a test file...")
    
    # Create a simple test restaurant file
    from datetime import datetime
    test_data = [
        {
            "name": "Test Restaurant",
            "cuisine": "Indian",
            "address": "Roorkee City Center",
            "status": "open",
            "phone": "+91 1234567890",
            "ratings": {"Dining": 4.5},
            "menu_items": {
                "Main Course": [
                    {"name": "Paneer Butter Masala", "price": "₹250", "type": "veg", "description": "Cottage cheese in rich tomato gravy"}
                ],
                "Desserts": [
                    {"name": "Gulab Jamun", "price": "₹100", "type": "veg", "description": "Sweet dumpling"}
                ]
            },
            "source": "test",
            "scraped_at": datetime.now().isoformat(),
            "url": "https://example.com/test-restaurant"
        }
    ]

    # Save to file
    test_file = f"restaurants_test_{datetime.now().strftime('%Y%m%d')}.json"
    test_path = os.path.join(os.getcwd(), test_file)
    
    try:
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Created test file with sample data: {test_path}")
        return test_path
    except Exception as e:
        logger.error(f"Failed to create test file: {e}")
        
    raise FileNotFoundError("No restaurant JSON files found. Run the scraper first.")

class RestaurantRAG:
    """
    Retrieval Augmented Generation system for restaurant queries based on restaurant data.
    
    This system uses a combination of vector search and language models to provide
    accurate answers to questions about restaurants and their menus.
    """
    
    def __init__(self, json_file_path: str = None, use_lightweight: bool = True, clear_cache: bool = False):
        """
        Initialize the RAG system with restaurant data.
        
        Args:
            json_file_path: Path to the restaurant JSON file (will find latest if None)
            use_lightweight: Use lightweight models to save memory
            clear_cache: Force recreation of vector store cache
        """
        if json_file_path is None:
            json_file_path = find_latest_json_file()
            logger.info(f"Using latest restaurant data file: {json_file_path}")
        
        logger.info("Initializing Restaurant RAG system...")
        
        # Step 1: Load restaurant knowledge base
        logger.info("Loading restaurant knowledge base...")
        from knowledge_base import RestaurantKnowledgeBase
        self.kb = RestaurantKnowledgeBase()
        self.kb.load_from_json(json_file_path)
        
        # Step 2: Create embeddings model
        self.embeddings = RestaurantEmbeddings.create_embeddings(use_lightweight)
        
        # Step 3: Set up document store
        self.document_store = RestaurantDocumentStore(
            self.kb, 
            self.embeddings,
            json_file_path,
            clear_cache
        )
        
        # Step 4: Initialize query engine
        self.query_engine = RestaurantQueryEngine(
            self.kb,
            self.document_store,
            use_lightweight
        )
        
    def query(self, question: str) -> Dict[str, Union[str, float]]:
        """
        Query the RAG system and get a response with metadata.
        
        Args:
            question: The user's question about restaurants or dishes
            
        Returns:
            Dictionary with answer and metadata like processing time and query type
        """
        return self.query_engine.query(question)
    
    def get_restaurant_info(self, name: str) -> Dict:
        """Get detailed information about a specific restaurant."""
        return self.query_engine.get_restaurant_info(name)
    
    def search_dishes(self, query: str) -> List[Dict]:
        """Search for dishes across all restaurants."""
        return self.query_engine.search_dishes(query)

def demo_rag():
    """Run a demo of the Restaurant RAG system."""
    # Find the latest JSON file
    try:
        json_file = find_latest_json_file()
        logger.info(f"Using data from: {json_file}")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Running the knowledge base to create the data...")
        
        # Import and run the knowledge base
        try:
            from knowledge_base import main as kb_main
            kb_main()
        except Exception as kb_error:
            logger.error(f"Failed to run knowledge base: {kb_error}")
            return
        
        # Try finding the file again
        json_file = find_latest_json_file()
        logger.info(f"Created and using data from: {json_file}")
    
    # Check if we should use lightweight mode
    use_lightweight = input("Use lightweight model to save memory? (y/n, default: y): ").lower() != 'n'
    clear_cache = input("Clear vector cache and rebuild? (y/n, default: n): ").lower() == 'y'
    
    # Create the RAG system
    try:
        rag = RestaurantRAG(json_file, use_lightweight=use_lightweight, clear_cache=clear_cache)
    except Exception as e:
        logger.error(f"Error initializing RAG with standard settings: {e}")
        logger.info("Falling back to lightweight mode...")
        rag = RestaurantRAG(json_file, use_lightweight=True)
    
    # Demo queries tailored for Roorkee restaurants
    demo_questions = [
        "What North Indian restaurants are there in Roorkee?",
        "Where can I find good vegetarian food in Roorkee?",
        "Are there any bakeries in Roorkee?",
        "Which restaurants serve chole bhature in Roorkee?",
        "What's the price range for pizzas in Roorkee?",
        "Where can I find paneer dishes in Roorkee?"
    ]
    
    print("\n" + "="*60)
    print("Restaurant RAG Demo for Roorkee, India")
    print("="*60)
    
    # Run the demo queries
    for i, question in enumerate(demo_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 50)
        try:
            result = rag.query(question)
            print(f"Answer: {result['answer']}")
            print(f"Query type: {result['query_type']} (processed in {result['processing_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
        print("-" * 50)
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode (type 'exit' to quit)")
    print("="*60)
    print("Ask questions about restaurants in Roorkee!")
    
    while True:
        user_question = input("\nYour question: ")
        if user_question.lower() in ['exit', 'quit', 'q']:
            break
        
        try:
            result = rag.query(user_question)
            print(f"\nAnswer: {result['answer']}")
            print(f"Query type: {result['query_type']} (processed in {result['processing_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error: {str(e)}")