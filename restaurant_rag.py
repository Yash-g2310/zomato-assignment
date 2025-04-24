import os
import json
import time
from typing import List, Dict, Any, Set, Union
from tqdm import tqdm
import torch
import gc
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Vector store and embeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# LLM
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# LangChain components
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser

# Restaurant data
from knowledge_base import RestaurantKnowledgeBase

class RestaurantRAG:
    """
    Retrieval Augmented Generation system for restaurant queries based on Zomato data.
    
    This system uses a combination of vector search and language models to provide
    accurate answers to questions about restaurants and their menus.
    """
    
    def __init__(self, json_file_path: str, use_lightweight: bool = True, clear_cache: bool = False):
        """
        Initialize the RAG system with restaurant data.
        
        Args:
            json_file_path: Path to the Zomato JSON file
            use_lightweight: Use lightweight models to save memory
            clear_cache: Force recreation of vector store cache
        """
        logger.info("Initializing Restaurant RAG system...")
        
        # Step 1: Load restaurant knowledge base
        logger.info("Loading restaurant knowledge base...")
        self.kb = RestaurantKnowledgeBase()
        self.kb.load_from_json(json_file_path)
        
        # Step 2: Create embeddings model
        logger.info("Setting up embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2" if use_lightweight 
                      else "sentence-transformers/all-mpnet-base-v2"
        )
        
        # Step 3: Set up vector stores
        cache_dir = "vector_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Delete cache if requested
        if clear_cache:
            import shutil
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.unlink(item_path)
            logger.info("Vector cache cleared")
            
        logger.info("Setting up vector stores...")
        self._setup_vector_stores(json_file_path, cache_dir)
        
        # Step 4: Initialize language model
        logger.info("Loading language model (this may take a moment)...")
        self._setup_llm(use_lightweight)
        
        # Step 5: Set up RAG pipelines
        logger.info("Creating RAG query chains...")
        self._setup_rag_pipelines()
        
        # Free up memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"Restaurant RAG system ready with {len(self.restaurant_docs)} restaurant documents "
                   f"and {len(self.dish_docs)} dish documents")

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
    
    def _setup_llm(self, use_lightweight: bool = True):
        """Set up a Hugging Face language model optimized for question answering."""
        try:
            # Check GPU availability
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                device_info = torch.cuda.get_device_properties(0)
                logger.info(f"Using GPU: {device_info.name} with {device_info.total_memory / 1e9:.2f} GB memory")
            else:
                logger.info("No GPU detected, using CPU (this will be slower)")
            
            # Choose model based on resource constraints
            if use_lightweight:
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                logger.info(f"Using lightweight model: {model_id}")
                
                # For GPU with RTX 4060 we can use 4-bit quantization
                from transformers import BitsAndBytesConfig
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # If GPU is available, use quantization
                if has_cuda:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        quantization_config=quantization_config
                    )
                else:
                    # CPU fallback
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        device_map=None
                    )
            else:
                # With RTX 4060 we can use a more powerful model
                model_id = "microsoft/phi-2"
                logger.info(f"Using standard model: {model_id}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # If GPU is available, use it
                if has_cuda:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    # CPU fallback
                    model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float32,
                        device_map=None
                    )
                
            # Create text generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
            
            # Wrap the pipeline in LangChain
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Falling back to CPU mode...")
            
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map=None,  # Force CPU
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,  # Reduce for CPU
                temperature=0.1,
                top_p=0.95,
                repetition_penalty=1.15,
                do_sample=True
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def _setup_rag_pipelines(self):
        """Set up the RAG pipelines with proper hybrid search."""
        # Restaurant query template
        restaurant_template = """You are a helpful restaurant assistant with knowledge about restaurants in Roorkee, India.
Answer the user's question based only on the following restaurant information:

{context}

User question: {question}

Answer in a helpful, concise way using only the information provided above.
If multiple restaurants match the query, list 2-3 options with their key details.
If the information doesn't contain the answer, say "I don't have that information."
"""
        restaurant_prompt = ChatPromptTemplate.from_template(restaurant_template)
        
        # Dish query template
        dish_template = """You are a helpful restaurant menu specialist with knowledge about dishes served in Roorkee, India.
Answer the user's question based only on the following dish information:

{context}

User question: {question}

Answer in a helpful, concise way using only the information provided above.
If many dishes match, focus on the most relevant ones and include their prices.
If the information doesn't contain the answer, say "I don't have that information."
"""
        dish_prompt = ChatPromptTemplate.from_template(dish_template)
        
        # Set up the hybrid retriever function for restaurants
        def retrieve_restaurant_docs(query: str) -> List[Document]:
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
        
        # Set up the hybrid retriever function for dishes
        def retrieve_dish_docs(query: str) -> List[Document]:
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
        
        # Create the restaurant chain
        self.restaurant_chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in retrieve_restaurant_docs(x)]), 
             "question": RunnablePassthrough()}
            | restaurant_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Create the dish chain
        self.dish_chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in retrieve_dish_docs(x)]), 
             "question": RunnablePassthrough()}
            | dish_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> Dict[str, Union[str, float]]:
        """
        Query the RAG system and get a response with metadata.
        
        Args:
            question: The user's question about restaurants or dishes
            
        Returns:
            Dictionary with answer and metadata like processing time and query type
        """
        start_time = time.time()
        
        # Improved query classification
        dish_keywords = [
            "dish", "food", "menu", "eat", "serve", "cuisine", "meal", 
            "paneer", "chicken", "pizza", "vegetarian", "vegan", "chole", "bhature",
            "noodles", "chawal", "cake", "sweet", "rice", "dal"
        ]
        
        restaurant_keywords = [
            "restaurant", "place", "open", "rating", "address", "where", "location",
            "roorkee", "civil lines", "ganeshpur"
        ]
        
        price_keywords = ["price", "cost", "cheap", "expensive", "affordable", "budget", "₹"]
        
        # Count keyword matches
        dish_score = sum(1 for kw in dish_keywords if kw in question.lower())
        restaurant_score = sum(1 for kw in restaurant_keywords if kw in question.lower())
        has_price_keywords = any(kw in question.lower() for kw in price_keywords)
        
        # Process the query
        try:
            # Determine query type and process accordingly
            if has_price_keywords or (dish_score > 0 and restaurant_score > 0):
                # Questions about prices or complex queries should use both
                logger.info("Processing as combined query (restaurant + dishes)...")
                rest_answer = self.restaurant_chain.invoke(question)
                dish_answer = self.dish_chain.invoke(question)
                
                if "I don't have that information" in rest_answer and "I don't have that information" in dish_answer:
                    answer = "I don't have that specific information in my knowledge base."
                    query_type = "combined (no information)"
                elif "I don't have that information" in rest_answer:
                    answer = dish_answer
                    query_type = "combined (dish only)"
                elif "I don't have that information" in dish_answer:
                    answer = rest_answer
                    query_type = "combined (restaurant only)"
                else:
                    # Combine the answers in a meaningful way
                    answer = f"{rest_answer}\n\nAdditional dish information: {dish_answer}"
                    query_type = "combined"
            elif dish_score > restaurant_score:
                logger.info("Processing as dish query...")
                answer = self.dish_chain.invoke(question)
                query_type = "dish"
            else:
                logger.info("Processing as restaurant query...")
                answer = self.restaurant_chain.invoke(question)
                query_type = "restaurant"
                
            # Clean up the response - remove model-specific formatting
            answer = answer.replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()
            
            # Clean up any trailing text that might come from the model
            if "<" in answer and ">" in answer:
                # Try to cut at the first tag after the main content
                main_content = answer.split("<")[0].strip()
                if main_content:
                    answer = main_content
        
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            answer = f"I'm sorry, but I encountered an error processing your question. Please try again or rephrase your question."
            query_type = "error"
        
        # Calculate response time
        elapsed_time = time.time() - start_time
        logger.info(f"Query processed in {elapsed_time:.2f} seconds")
        
        # Return both the answer and metadata
        return {
            "answer": answer,
            "query_type": query_type,
            "processing_time": elapsed_time
        }
    
    def get_restaurant_info(self, name: str) -> Dict:
        """Get detailed information about a specific restaurant."""
        restaurant = self.kb.get_restaurant(name)
        if restaurant:
            info = restaurant.get_basic_info()
            price_range = restaurant.get_price_range()
            
            # Get top dishes
            dishes = []
            for category, items in restaurant.menu_items.items():
                for item in items[:3]:  # Just take a few from each category
                    dishes.append({
                        "name": item.get("name", "Unknown"),
                        "price": item.get("price", "Price not available"),
                        "type": item.get("type", "unknown"),
                        "category": category
                    })
            
            return {
                "name": name,
                "info": info,
                "price_range": price_range,
                "sample_dishes": dishes[:5]  # Limit to 5 dishes total
            }
        return None
    
    def search_dishes(self, query: str) -> List[Dict]:
        """Search for dishes across all restaurants."""
        results = []
        for name, restaurant in self.kb.restaurants.items():
            dishes = restaurant.find_dishes(query.lower())
            for dish in dishes:
                results.append({
                    "dish_name": dish.get("name", "Unknown"),
                    "price": dish.get("price", "Price not available"),
                    "restaurant": name,
                    "category": dish.get("category", ""),
                    "type": dish.get("type", "unknown")
                })
        
        return sorted(results, key=lambda x: x["dish_name"])[:10]  # Return top 10 matches


def find_latest_json_file():
    """Find the latest Zomato JSON file in the current directory or parent."""
    from config import get_latest_data_file
     # First try to find combined data files
    latest = get_latest_data_file(prefix="restaurants_")
    if latest:
        return latest
        
    # Fall back to Zomato-only files
    latest = get_latest_data_file(prefix="zomato_")
    if latest:
        return latest
        
    # Fall back to Swiggy-only files
    latest = get_latest_data_file(prefix="swiggy_")
    if latest:
        return latest
        
    raise FileNotFoundError("No restaurant JSON files found. Run the scraper first.")


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


if __name__ == "__main__":
    demo_rag()