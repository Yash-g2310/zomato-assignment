import os
import json
import shutil
import logging
from typing import List, Dict, Any, Set, Union
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

class RestaurantDocumentStore:
    """
    Handles document creation and vector stores for restaurant data.
    """
    
    def __init__(self, kb, embeddings, json_file_path: str, clear_cache: bool = False):
        """
        Initialize the document store.
        
        Args:
            kb: Restaurant knowledge base
            embeddings: Embedding model to use
            json_file_path: Path to the JSON data file
            clear_cache: Whether to clear existing cache
        """
        self.kb = kb
        self.embeddings = embeddings
        self.json_file_path = json_file_path
        
        # Set up vector stores
        cache_dir = "vector_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Delete cache if requested
        if clear_cache:
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.unlink(item_path)
            logger.info("Vector cache cleared")
            
        logger.info("Setting up document store...")
        self._setup_vector_stores(json_file_path, cache_dir)
        
    def _setup_vector_stores(self, json_file_path: str, cache_dir: str):
        """Set up vector stores with caching for restaurants and dishes."""
        # Generate cache paths
        json_basename = os.path.basename(json_file_path).split('.')[0]
        restaurant_cache = os.path.join(cache_dir, f"{json_basename}_restaurants")
        dish_cache = os.path.join(cache_dir, f"{json_basename}_dishes")
        
        # Process restaurant data
        self.restaurant_docs = []
        for name, restaurant in self.kb.restaurants.items():
            self.restaurant_docs.append(self._create_restaurant_document(name, restaurant))
            
        # Process dish data first to have it available for both scenarios
        self.dish_docs = []
        for restaurant_name, restaurant in self.kb.restaurants.items():
            self.dish_docs.extend(self._create_dish_documents_for_restaurant(restaurant_name, restaurant))
        
        # Create or load restaurant vectors
        try:
            if os.path.exists(restaurant_cache):
                logger.info("Loading cached restaurant embeddings...")
                self.restaurant_vectorstore = FAISS.load_local(
                    restaurant_cache, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating restaurant embeddings...")
                self.restaurant_vectorstore = FAISS.from_documents(
                    self.restaurant_docs, self.embeddings
                )
                self.restaurant_vectorstore.save_local(restaurant_cache)
                
        except Exception as e:
            logger.error(f"Error loading restaurant vectors: {e}")
            logger.info("Creating fresh restaurant embeddings...")
            self.restaurant_vectorstore = FAISS.from_documents(
                self.restaurant_docs, self.embeddings
            )
            self.restaurant_vectorstore.save_local(restaurant_cache)
        
        # Create or load dish vectors
        try:
            if os.path.exists(dish_cache):
                logger.info("Loading cached dish embeddings...")
                self.dish_vectorstore = FAISS.load_local(
                    dish_cache, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            else:
                logger.info("Creating dish embeddings...")
                self.dish_vectorstore = FAISS.from_documents(
                    self.dish_docs, self.embeddings
                )
                self.dish_vectorstore.save_local(dish_cache)
                
        except Exception as e:
            logger.error(f"Error loading dish vectors: {e}")
            logger.info("Creating fresh dish embeddings...")
            self.dish_vectorstore = FAISS.from_documents(
                self.dish_docs, self.embeddings
            )
            self.dish_vectorstore.save_local(dish_cache)
            
        # Set up BM25 retrievers for hybrid search
        self.restaurant_bm25 = BM25Retriever.from_documents(self.restaurant_docs)
        self.restaurant_bm25.k = 5
        
        self.dish_bm25 = BM25Retriever.from_documents(self.dish_docs)
        self.dish_bm25.k = 7
    
    def _create_restaurant_document(self, name: str, restaurant) -> Document:
        """Create a document for a restaurant with all relevant details."""
        info = restaurant.get_basic_info()
        price_range = restaurant.get_price_range()
        
        # Create structured content for better retrieval
        content = f"""
        Restaurant: {name}
        Cuisine: {info['cuisine']}
        Address: {info['address']}
        Status: {info['status']}
        Phone: {info['phone']}
        Ratings: {json.dumps(info['ratings']) if info['ratings'] else 'No ratings available'}
        Dish Count: {info['dish_count']}
        Price Range: ₹{price_range['min']} - ₹{price_range['max']} (Average: ₹{price_range['avg']:.2f})
        Has Vegetarian Options: {"Yes" if info['has_veg_options'] else "No"}
        Menu Categories: {', '.join(restaurant.dish_categories) if restaurant.dish_categories else 'No categories'}
        """
        
        # Add metadata for filtering
        return Document(
            page_content=content,
            metadata={
                "name": name,
                "cuisine": info['cuisine'],
                "status": info['status'],
                "address": info['address'],
                "min_price": price_range['min'],
                "max_price": price_range['max'],
                "avg_price": price_range['avg'],
                "type": "restaurant"  # Add document type for filtering
            }
        )
    
    def _create_dish_documents_for_restaurant(self, restaurant_name: str, restaurant) -> List[Document]:
        """Create document objects for menu items with deduplication."""
        documents = []
        seen_dishes: Set[str] = set()  # Track unique dishes
        
        for category, items in restaurant.menu_items.items():
            for item in items:
                name = item.get('name', '')
                if not name:
                    continue
                
                # Create a unique identifier for the dish
                dish_id = f"{name}_{item.get('price', '')}"
                
                # Skip duplicates
                if dish_id in seen_dishes:
                    continue
                
                seen_dishes.add(dish_id)
                
                # Create structured content for each dish
                content = f"""
                Dish: {name}
                Restaurant: {restaurant_name}
                Category: {category}
                Price: {item.get('price', 'Price not available')}
                Description: {item.get('description', 'No description available')}
                Type: {item.get('type', 'unknown')}
                """
                
                # Add to documents with metadata
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "dish_name": name,
                            "restaurant": restaurant_name,
                            "category": category,
                            "price": item.get('price', 'Price not available'),
                            "type": item.get('type', 'unknown'),
                            "doc_type": "dish"  # Add document type for filtering
                        }
                    )
                )
        
        return documents
    
    def get_hybrid_restaurant_documents(self, query: str) -> List[Document]:
        """Retrieve restaurant documents using hybrid search."""
        # Get semantic search results
        semantic_docs = self.restaurant_vectorstore.similarity_search(query, k=5)
        
        # Get keyword search results
        keyword_docs = self.restaurant_bm25.get_relevant_documents(query)
        
        # Combine results with deduplication
        seen_content = set()
        hybrid_docs = []
        
        for doc in semantic_docs + keyword_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                hybrid_docs.append(doc)
        
        return hybrid_docs[:5]  # Return top 5 unique documents
    
    def get_hybrid_dish_documents(self, query: str) -> List[Document]:
        """Retrieve dish documents using hybrid search."""
        # Get semantic search results
        semantic_docs = self.dish_vectorstore.similarity_search(query, k=7)
        
        # Get keyword search results
        keyword_docs = self.dish_bm25.get_relevant_documents(query)
        
        # Combine results with deduplication
        seen_content = set()
        hybrid_docs = []
        
        for doc in semantic_docs + keyword_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                hybrid_docs.append(doc)
        
        return hybrid_docs[:7]  # Return top 7 unique documents