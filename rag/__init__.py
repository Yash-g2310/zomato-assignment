"""
Restaurant RAG (Retrieval Augmented Generation) Package

This package provides components for creating a RAG system for restaurant data,
enabling natural language queries about restaurants and their menus.
"""

import os
import logging
import json
import glob
import time
import re

from typing import Dict, List, Union, Any
from .embeddings import create_embedding_model
from .document_store import RestaurantDocumentStore
from .reranker import QueryDocumentReranker
from .query_engine import RestaurantQueryEngine
from .llm import create_llm
from config import get_latest_data_file
from datetime import datetime
from config import get_latest_data_file
from knowledge_base import RestaurantKnowledgeBase, KnowledgeBaseManager
from knowledge_base import main as kb_main

logger = logging.getLogger(__name__)

def find_latest_json_file():
    """Find the latest restaurant JSON file in the current directory or parent."""
    
    
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

def find_all_restaurant_files():
    """Find all restaurant data files."""
    
    all_files = []
    
    # Find all restaurant data files
    for prefix in ["restaurants_", "zomato_", "swiggy_"]:
        pattern = os.path.join(os.getcwd(), f"{prefix}*.json")
        files = glob.glob(pattern)
        if files:
            all_files.extend(files)
    
    if not all_files:
        # Fall back to finding one latest file
        try:
            latest = find_latest_json_file()
            all_files = [latest]
        except FileNotFoundError:
            raise FileNotFoundError("No restaurant JSON files found. Run the scraper first.")
    
    logger.info(f"Found {len(all_files)} restaurant data files")
    return all_files

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
        
        logger.info("Initializing Restaurant RAG system...")
        self.kb_manager = KnowledgeBaseManager()
        
        # Step 1: Load restaurant knowledge base
        logger.info("Loading restaurant knowledge base...")
        
        all_patterns = [
            "restaurants_*.json", 
            "zomato_*.json",       
            "swiggy_*.json",
            "*.json"  # Fallback to ensure all JSON files are checked
        ]
        
        if json_file_path:
            logger.info(f"Using specific restaurant data file: {json_file_path}")
            self.kb_manager.update_from_json_files([json_file_path])
        else:
            logger.info("Loading from all restaurant data files...")
            self.kb_manager.update_from_json_files(all_patterns)
            
        
        
        # Access KB from manager
        self.kb = self.kb_manager.kb
        
        logger.info(f"Loaded {len(self.kb.restaurants)} restaurants into knowledge base")

        # Step 2: Create embeddings model
        self.embeddings = create_embedding_model(use_lightweight)

        # Step 3: Set up document store
        self.document_store = RestaurantDocumentStore(
            self.kb, 
            self.embeddings,
            json_file_path,
            clear_cache
        )

        # Step 4: Initialize reranker for better retrieval precision
        
        self.reranker = QueryDocumentReranker(use_lightweight)

        # Step 5: Initialize LLM (try improved approach first)
        self.query_engine = RestaurantQueryEngine(
            self.kb,
            self.document_store,
            use_lightweight
        )
        try:
            
            self.llm = create_llm(use_lightweight)
            if not self.llm:
                self.llm = self.query_engine.llm
        except ImportError:
            self.llm = self.query_engine.llm

        # Set up query cache
        self.query_cache = {}
        
    def query(self, user_question: str, use_cache: bool = True) -> dict:
        """
        Process a natural language question about restaurants.

        Args:
            user_question: The user's question
            use_cache: Whether to use cached results

        Returns:
            A dictionary with the answer and metadata
        """
        start_time = time.time()

        # Check cache if enabled
        if use_cache and self.query_cache is not None and user_question in self.query_cache:
            cached_result = self.query_cache[user_question]
            cached_result["source"] = "cache"
            cached_result["processing_time"] = time.time() - start_time
            return cached_result

        # Try using query_engine first if it exists and has a query method
        if hasattr(self, 'query_engine') and hasattr(self.query_engine, 'query'):
            try:
                result = self.query_engine.query(user_question)
                # Add metadata
                if "processing_time" not in result:
                    result["processing_time"] = time.time() - start_time

                # Update cache
                if use_cache and self.query_cache is not None:
                    self.query_cache[user_question] = result

                return result
            except Exception as e:
                logger.warning(f"Query engine failed: {e}, falling back to direct query handling")

        # Fall back to specialized query handling
        # Create query plan
        query_plan = self._plan_query(user_question)

        # Handle comparison queries with specialized processing
        if query_plan["requires_comparison"] and len(query_plan["entities"]) >= 2:
            result = self._handle_comparison_query(query_plan)

        # Handle dietary restriction queries
        elif query_plan["query_type"] == "dietary_query":
            result = self._handle_dietary_query(query_plan)

        # Handle menu item queries
        elif query_plan["query_type"] == "menu_item_query":
            result = self._handle_menu_item_query(query_plan)

        # Handle price queries
        elif query_plan["query_type"] == "price_query":
            result = self._handle_price_query(query_plan)

        # Handle general queries with standard processing
        else:
            result = self._handle_general_query(query_plan)

        # Update cache
        if use_cache and self.query_cache is not None:
            self.query_cache[user_question] = result

        # Add processing time
        result["processing_time"] = time.time() - start_time
        result["query_type"] = query_plan["query_type"]

        return result
    

    def _identify_restriction(self, query: str) -> str:
        """Identify dietary restriction type from query."""
        query = query.lower()
        
        if "vegan" in query:
            return "vegan"
        elif "vegetarian" in query:
            return "vegetarian"
        elif "gluten" in query:
            return "gluten-free"
        elif "dairy" in query or "lactose" in query:
            return "dairy-free"
        elif "nut" in query or "peanut" in query:
            return "nut-free"
        elif "spicy" in query or "spice" in query:
            return "spicy"
        
        # Default to vegetarian as the most common restriction
        return "vegetarian"
    
    def _extract_food_item(self, query: str) -> str:
        """Extract the food item from a menu query."""
        query = query.lower()
        
        # Look for phrases like "serve X", "have X", "offer X"
        patterns = [
            r"(?:serve|have|offer|sell|got|menu has|provides)\s+([a-zA-Z\s]+)",
            r"(?:looking for|want|find|any)\s+([a-zA-Z\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                # Clean up the extracted item
                item = match.group(1).strip()
                # Remove trailing words often part of questions
                item = re.sub(r'\s+(?:in|at|on|from|with|and|or)\s+.*$', '', item)
                return item
        
        # Fall back to looking for menu items we know about
        known_items = set()
        for restaurant in self.kb.restaurants.values():
            for category in restaurant.menu_items.values():
                for item in category:
                    if 'name' in item:
                        known_items.add(item['name'].lower())
        
        # Find the longest item name that's in the query
        matches = [item for item in known_items if item in query]
        if matches:
            return max(matches, key=len)
        
        # Last resort: return the query without common question words
        cleaned = query.lower()
        for word in ["where", "can", "i", "find", "is", "there", "any", "do", "they", "have"]:
            cleaned = cleaned.replace(word, "")
        
        return cleaned.strip()
    
    def _handle_menu_item_query(self, query_plan):
        """Handle queries about specific menu items."""
        food_item = query_plan.get("food_item", "")
        original_query = query_plan["original_query"]
        
        # Search for dishes matching the food item
        dish_results = self.search_dishes(food_item)
        
        if not dish_results:
            # Fall back to general query if no dishes found
            return self._handle_general_query(query_plan)
        
        # Build context with the dish information
        context = f"Query about: {food_item}\n\n"
        context += f"Found {len(dish_results)} related dishes:\n\n"
        
        for i, dish in enumerate(dish_results[:5], 1):
            context += f"{i}. {dish['dish_name']} at {dish['restaurant']}\n"
            context += f"   Price: {dish['price']}\n"
            context += f"   Category: {dish['category']}\n"
            if dish.get('type'):
                context += f"   Type: {dish['type']}\n"
            if dish.get('description'):
                context += f"   Description: {dish['description']}\n"
            context += "\n"
        
        prompt = f"""
        Based on the restaurant information below, please answer the user's question.
        
        {context}
        
        User question: {query_plan["original_query"]}
        
        IMPORTANT: ONLY provide information about restaurants that are explicitly mentioned in the context above.
        DO NOT make up or hallucinate restaurant details not present in the provided information.
        If the specific restaurant the user asked about isn't in the context, say so clearly.
        
        Your answer:
        """
        
        answer = self.llm(prompt)
        
        return {
            "answer": answer,
            "sources": [dish["restaurant"] for dish in dish_results[:5]],
            "query_type": "menu_item_query",
            "food_item": food_item
        }
    
    def _handle_price_query(self, query_plan):
        """Handle queries about price ranges."""
        original_query = query_plan["original_query"]
        
        # Check if this is about a specific restaurant
        restaurant_name = None
        for name in self.kb.restaurants:
            if name.lower() in original_query.lower():
                restaurant_name = name
                break
            
        if restaurant_name:
            # Query about specific restaurant's prices
            restaurant_info = self.get_restaurant_info(restaurant_name)
            price_range = restaurant_info["price_range"]
            
            context = f"Restaurant: {restaurant_name}\n"
            context += f"Price range: ₹{price_range['min']} to ₹{price_range['max']} (average: ₹{price_range['avg']:.2f})\n\n"
            context += "Sample prices:\n"
            
            # Add sample dishes with prices, grouped by category
            by_category = {}
            for dish in restaurant_info.get("sample_dishes", []):
                category = dish.get("category", "Other")
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(dish)
            
            for category, dishes in by_category.items():
                context += f"\n{category}:\n"
                for dish in dishes[:3]:
                    context += f"- {dish['name']}: {dish['price']}\n"
        else:
            # General price query
            docs = self.document_store.get_hybrid_restaurant_documents("price range", self.reranker)
            restaurant_names = [doc.metadata["name"] for doc in docs]
            
            context = "Price information for restaurants:\n\n"
            
            for name in restaurant_names[:5]:
                restaurant_info = self.get_restaurant_info(name)
                info = restaurant_info["info"]
                price_range = restaurant_info["price_range"]
                
                context += f"Restaurant: {name}\n"
                context += f"Cuisine: {info['cuisine']}\n"
                context += f"Price range: ₹{price_range['min']} to ₹{price_range['max']} (average: ₹{price_range['avg']:.2f})\n"
                context += "Sample dishes:\n"
                
                # Get a few dishes with different prices
                dishes = restaurant_info.get("sample_dishes", [])
                if dishes:
                    # Try to get a low, medium and high priced item
                    dishes.sort(key=lambda x: x.get('price_numeric', 0))
                    samples = [dishes[0]]  # Lowest price
                    
                    if len(dishes) > 2:
                        samples.append(dishes[len(dishes)//2])  # Medium price
                    
                    if len(dishes) > 1:
                        samples.append(dishes[-1])  # Highest price
                    
                    for dish in samples:
                        context += f"- {dish['name']}: {dish['price']}\n"
                
                context += "\n"
        
        prompt = f"""
        Based on the restaurant information below, please answer the user's question.

        {context}

        User question: {query_plan["original_query"]}

        IMPORTANT: ONLY provide information about restaurants that are explicitly mentioned in the context above.
        DO NOT make up or hallucinate restaurant details not present in the provided information.
        If the specific restaurant the user asked about isn't in the context, say so clearly.

        Your answer:
        """
        
        answer = self.llm(prompt)
        
        return {
            "answer": answer,
            "sources": [restaurant_name] if restaurant_name else restaurant_names,
            "query_type": "price_query"
        }
    
    def _handle_general_query(self, query_plan: dict) -> dict:
        """Handle general queries about restaurants."""
        # Get relevant documents first
        docs = self.document_store.get_hybrid_restaurant_documents(query_plan["original_query"], self.reranker)

        # Build context from the documents
        context = "Information about relevant restaurants:\n\n"
        for doc in docs[:3]:
            context += doc.page_content + "\n\n"

        # Then use the context in your prompt
        prompt = f"""
        Based on the restaurant information below, please answer the user's question.

        {context}

        User question: {query_plan["original_query"]}

        IMPORTANT: ONLY provide information about restaurants that are explicitly mentioned in the context above.
        DO NOT make up or hallucinate restaurant details not present in the provided information.
        If the specific restaurant the user asked about isn't in the context, say so clearly.

        Your answer:
        """

        try:
            # Get the raw response from the LLM
            raw_answer = self.llm(prompt)

            # Clean up the response:
            # 1. Remove any part that might repeat the prompt
            if "Your answer:" in raw_answer:
                answer = raw_answer.split("Your answer:")[-1].strip()
            elif "User question:" in raw_answer:
                # Sometimes models repeat part of the prompt
                answer = raw_answer.split("User question:")[-1].strip()
                if answer.startswith(query_plan["original_query"]):
                    answer = answer[len(query_plan["original_query"]):].strip()
            else:
                answer = raw_answer.strip()
                # Remove any system instructions text that might have been included
                if "IMPORTANT:" in answer:
                    answer = answer.split("IMPORTANT:")[0].strip()

                # Return the cleaned answer
                return {
                    "answer": answer,
                    "sources": [doc.metadata.get("name", "Unknown") for doc in docs],
                    "query_type": "general_query"
                }
                
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            answer = "I apologize, but I'm currently having technical difficulties accessing the restaurant information."

        return {
            "answer": answer,
            "sources": [doc.metadata.get("name", "Unknown") for doc in docs],
            "query_type": "general_query"
        }
    
    def find_restaurant_by_name(self, name_query: str) -> str:
        """Find the closest restaurant name match."""
        name_query = name_query.lower()
        
        # First try exact matches
        for restaurant_name in self.kb.restaurants:
            if restaurant_name.lower() == name_query:
                return restaurant_name
        
        # Then try substring matches
        matching_restaurants = []
        for restaurant_name in self.kb.restaurants:
            if name_query in restaurant_name.lower() or restaurant_name.lower() in name_query:
                matching_restaurants.append((restaurant_name, 
                                           len(set(name_query.split()) & set(restaurant_name.lower().split()))))
        
        # Return the best match if any
        if matching_restaurants:
            # Sort by number of matching words
            matching_restaurants.sort(key=lambda x: x[1], reverse=True)
            return matching_restaurants[0][0]
        
        return None
    
    def _handle_comparison_query(self, query_plan: dict) -> dict:
        """Handle comparison between restaurants or menu items."""
        original_query = query_plan["original_query"]
        entities = query_plan["entities"]
        query_type = query_plan["query_type"]

        # Gather data for each entity
        entity_data = []
        for entity in entities:
            # Try to find the entity as a restaurant
            restaurant_match = self.find_restaurant_by_name(entity)
            if restaurant_match:
                entity_data.append({
                    "type": "restaurant",
                    "name": restaurant_match,
                    "data": self.get_restaurant_info(restaurant_match)
                })
            else:
                # Look for it as a dish/food item
                dish_results = self.search_dishes(entity)
                if dish_results:
                    entity_data.append({
                        "type": "dish",
                        "name": entity,
                        "data": dish_results
                    })

        # If we couldn't find data for all entities, fall back to general query
        if len(entity_data) < len(entities):
            return self._handle_general_query(query_plan)

        # Prepare context with the comparison data
        comparison_context = self._prepare_comparison_context(entity_data, query_type)

        # Generate the answer using the LLM with structured context
        prompt_template = """
        Based on the restaurant information below, please answer the user's question.

        {context}

        User question: {question}

        Please provide a detailed comparison addressing the user's specific question.
        If the information isn't available in the context, please say so.
        """

        prompt = prompt_template.format(
            context=comparison_context,
            question=original_query
        )

        # Get answer from LLM
        answer = self.llm(prompt)

        return {
            "answer": answer,
            "sources": [item["name"] for item in entity_data],
            "query_type": f"comparison_{query_type}"
        }

    def _prepare_comparison_context(self, entity_data: list, query_type: str) -> str:
        """Prepare structured context for comparison queries."""
        context = "COMPARISON INFORMATION:\n\n"

        # For price comparisons, focus on price data
        if query_type == "price_comparison":
            for item in entity_data:
                if item["type"] == "restaurant":
                    context += f"Restaurant: {item['name']}\n"
                    price_range = item["data"]["price_range"]
                    context += f"Price range: ₹{price_range['min']} to ₹{price_range['max']} (average: ₹{price_range['avg']:.2f})\n"

                    # Add some sample dishes with prices
                    context += "Sample menu items with prices:\n"
                    for dish in (item["data"]["sample_dishes"] or [])[:5]:
                        context += f"- {dish['name']}: {dish['price']}\n"
                    context += "\n"

        # For menu comparisons, focus on menu items
        elif query_type == "menu_comparison":
            for item in entity_data:
                if item["type"] == "restaurant":
                    context += f"Restaurant: {item['name']}\n"
                    context += f"Cuisine: {item['data']['info']['cuisine']}\n"

                    context += "Menu categories: "
                    categories = set()
                    for dish in (item["data"]["sample_dishes"] or []):
                        categories.add(dish['category'])
                    context += ", ".join(categories) + "\n"

                    context += "Sample dishes:\n"
                    for dish in (item["data"]["sample_dishes"] or [])[:10]:
                        context += f"- {dish['name']} ({dish['category']}): {dish['price']}"
                        if dish.get('type'):
                            context += f" [{dish['type']}]"
                        context += "\n"
                    context += "\n"

        # For general comparisons, include all data
        else:
            for item in entity_data:
                if item["type"] == "restaurant":
                    context += f"Restaurant: {item['name']}\n"
                    info = item["data"]["info"]
                    context += f"Cuisine: {info['cuisine']}\n"
                    context += f"Address: {info['address']}\n"
                    context += f"Status: {info['status']}\n"
                    context += f"Ratings: {json.dumps(info['ratings'])}\n"
                    context += f"Price range: ₹{item['data']['price_range']['min']} to ₹{item['data']['price_range']['max']}\n\n"
                elif item["type"] == "dish":
                    context += f"Dish category: {item['name']}\n"
                    context += "Available at restaurants:\n"
                    for dish in item["data"][:5]:
                        context += f"- {dish['restaurant']}: {dish['dish_name']} - {dish['price']}\n"
                    context += "\n"

        return context

    def _handle_dietary_query(self, query_plan: dict) -> dict:
        """Handle queries about dietary restrictions."""
        # Extract dietary restriction type
        diet_type = query_plan.get("restriction_type", "vegetarian")  # Default to vegetarian if not specified

        # Is this about a specific restaurant?
        restaurant_name = None
        for name in self.kb.restaurants:
            if name.lower() in query_plan["original_query"].lower():
                restaurant_name = name
                break
            
        if restaurant_name:
            # Query about specific restaurant's dietary options
            restaurant_info = self.get_restaurant_info(restaurant_name)

            # Filter dishes by dietary restriction
            matching_dishes = []
            all_dishes = restaurant_info.get("sample_dishes", [])

            for dish in all_dishes:
                dish_type = dish.get("type", "").lower()
                dish_name = dish.get("name", "").lower()
                dish_desc = dish.get("description", "").lower()

                # Match dishes based on restriction type
                if diet_type == "vegetarian" and dish_type == "veg":
                    matching_dishes.append(dish)
                elif diet_type == "vegan" and ("vegan" in dish_name or "vegan" in dish_desc):
                    matching_dishes.append(dish)
                elif diet_type == "gluten-free" and ("gluten-free" in dish_name or "gluten-free" in dish_desc):
                    matching_dishes.append(dish)

            # Generate response with structured data
            context = f"Restaurant: {restaurant_name}\n"
            context += f"Dietary restriction: {diet_type}\n\n"

            if matching_dishes:
                context += f"Found {len(matching_dishes)} {diet_type} dishes:\n"
                for dish in matching_dishes:
                    context += f"- {dish['name']} ({dish['category']}): {dish['price']}\n"
            else:
                context += f"No specific {diet_type} dishes identified in the menu data.\n"

            # Generate answer using LLM
            prompt = f"""
            Based on the restaurant information below, please answer the user's question.

            {context}

            User question: {query_plan["original_query"]}

            IMPORTANT: ONLY provide information about restaurants that are explicitly mentioned in the context above.
            DO NOT make up or hallucinate restaurant details not present in the provided information.
            If the specific restaurant the user asked about isn't in the context, say so clearly.

            Your answer:
            """

            answer = self.llm(prompt)

            return {
                "answer": answer,
                "sources": [restaurant_name],
                "query_type": "dietary_query"
            }
        else:
            # General query about restaurants with dietary options
            # Use hybrid search to find restaurants with these options
            docs = self.document_store.get_hybrid_restaurant_documents(f"{diet_type} options", self.reranker)
            restaurant_names = [doc.metadata["name"] for doc in docs]

            context = f"Dietary restriction: {diet_type}\n\n"
            context += "Restaurants that might have suitable options:\n"

            for name in restaurant_names[:5]:
                restaurant_info = self.get_restaurant_info(name)
                if not restaurant_info.get("info"):
                    continue
                info = restaurant_info["info"]

                context += f"\nRestaurant: {name}\n"
                context += f"Cuisine: {info.get('cuisine', 'Not specified')}\n"
                context += f"Status: {info.get('status', 'Unknown')}\n"

                # Count matching dishes
                matching_count = 0
                for dish in restaurant_info.get("sample_dishes", []):
                    if diet_type == "vegetarian" and dish.get("type") == "veg":
                        matching_count += 1
                    elif diet_type.lower() in dish.get("name", "").lower() or diet_type.lower() in dish.get("description", "").lower():
                        matching_count += 1

                context += f"Matching dishes: approximately {matching_count}\n"

                # Add 2-3 example dishes
                matching_dishes = []
                for dish in restaurant_info.get("sample_dishes", []):
                    if diet_type == "vegetarian" and dish.get("type") == "veg":
                        matching_dishes.append(dish)
                    elif diet_type.lower() in dish.get("name", "").lower() or diet_type.lower() in dish.get("description", "").lower():
                        matching_dishes.append(dish)

                if matching_dishes:
                    context += "Example dishes:\n"
                    for dish in matching_dishes[:3]:
                        context += f"- {dish['name']}: {dish['price']}\n"

            prompt = f"""
            Based on the restaurant information below, please answer the user's question.

            {context}

            User question: {query_plan["original_query"]}

            IMPORTANT: ONLY provide information about restaurants that are explicitly mentioned in the context above.
            DO NOT make up or hallucinate restaurant details not present in the provided information.
            If the specific restaurant the user asked about isn't in the context, say so clearly.

            Your answer:
            """

            answer = self.llm(prompt)

            return {
                "answer": answer,
                "sources": restaurant_names,
                "query_type": "dietary_query"
            }
    
    
    
    # Replace the custom implementation with:

    def get_restaurant_info(self, name: str) -> Dict:
        """Get detailed information about a specific restaurant."""
        restaurant = self.kb.get_restaurant(name)
        if not restaurant:
            return {
                "info": {},
                "price_range": {"min": 0, "max": 0, "avg": 0},
                "sample_dishes": []
            }

        # Get data directly from the Restaurant object
        return {
            "info": restaurant.get_basic_info(),
            "price_range": restaurant.get_price_range(),
            "sample_dishes": [
                {
                    "name": item.get("name", "Unknown"),
                    "price": item.get("price", "Price not available"),
                    "category": category,
                    "type": item.get("type", "unknown"),
                    "description": item.get("description", "")
                }
                for category, items in list(restaurant.menu_items.items())[:3]
                for item in items[:2]
            ]
        }
    def _extract_price_numeric(self, price_str: str) -> int:
        """Extract numeric price value from price string."""
        if not price_str:
            return 0

        price_match = re.search(r'(\d+)', price_str)
        if price_match:
            return int(price_match.group(1))
        return 0
    
    def search_dishes(self, query: str) -> List[Dict]:
        """Search for dishes across all restaurants using the knowledge base directly."""
        # Use the KB's dish finding functionality 
        restaurants_with_dishes = self.kb.find_restaurants_with_dish(query)

        results = []
        for restaurant_result in restaurants_with_dishes:
            restaurant_name = restaurant_result["restaurant_name"]
            for dish in restaurant_result["matching_dishes"]:
                results.append({
                    "dish_name": dish.get("name", "Unknown dish"),
                    "restaurant": restaurant_name,
                    "category": dish.get("category", ""),
                    "price": dish.get("price", "Price not available"),
                    "price_numeric": self._extract_price_numeric(dish.get("price", "")),
                    "type": dish.get("type", "unknown"),
                    "description": dish.get("description", "")
                })

        return results

    def _plan_query(self, query: str) -> dict:
        """Create a query plan based on the input question."""
        query_type = self._detect_query_type(query)
        
        # Create a structured plan based on query type
        query_plan = {
            "original_query": query,
            "query_type": query_type,
            "requires_comparison": "compare" in query_type or "vs" in query or " or " in query,
            "processed_query": query,
        }
        
        # For comparison queries, extract the entities being compared
        if query_plan["requires_comparison"]:
            query_plan["entities"] = self._extract_comparison_entities(query)
            
        # For specific menu items, extract the food item
        if query_type == "menu_item_query":
            query_plan["food_item"] = self._extract_food_item(query)
            
        # For dietary restrictions, identify the restriction type
        if query_type == "dietary_query":
            query_plan["restriction_type"] = self._identify_restriction(query)
        
        return query_plan
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query for specialized handling."""
        query = query.lower()
        
        # Check for comparison queries
        if any(word in query for word in ["compare", "comparison", "versus", "vs", " or ", "better", "best", "difference"]):
            if any(word in query for word in ["price", "cost", "expensive", "cheap"]):
                return "price_comparison"
            if any(word in query for word in ["menu", "dish", "food", "item", "cuisine"]):
                return "menu_comparison"
            return "general_comparison"
        
        # Check for menu item queries
        if any(word in query for word in ["have", "offer", "serve", "menu", "dish", "food item"]):
            return "menu_item_query"
        
        # Check for dietary restriction queries
        if any(word in query for word in ["vegetarian", "vegan", "gluten", "dairy", "allergy", "allergic", "spicy"]):
            return "dietary_query"
        
        # Check for price queries
        if any(word in query for word in ["price", "cost", "expensive", "cheap", "budget", "affordable"]):
            return "price_query"
        
        # Default to general query
        return "general_query"
    
    def _extract_comparison_entities(self, query: str) -> list:
        """Extract entities being compared in comparison queries."""
        # This could be enhanced with more sophisticated NLP parsing
        
        # Look for common patterns
        patterns = [
            r"(?:compare|comparison|versus|vs|or|and|between)\s+([A-Za-z\s]+)\s+(?:and|or|to|with|versus|vs)\s+([A-Za-z\s]+)",
            r"(?:which is|what's|whats|what is)\s+(?:better|best|worse|worst)\s+([A-Za-z\s]+)\s+(?:or|vs|versus)\s+([A-Za-z\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return [match.group(1).strip(), match.group(2).strip()]
        
        # If no match, look for mentions of restaurant names
        restaurants = []
        for name in self.kb.restaurants:
            if name.lower() in query.lower():
                restaurants.append(name)
        
        return restaurants if restaurants else []

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