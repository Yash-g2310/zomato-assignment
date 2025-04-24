import time
import torch
import logging
from typing import Dict, List, Union, Any

from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

logger = logging.getLogger(__name__)

class RestaurantQueryEngine:
    """
    Query engine for restaurant data using RAG.
    """
    
    def __init__(self, kb, document_store, use_lightweight: bool = True):
        """
        Initialize the query engine.
        
        Args:
            kb: Restaurant knowledge base
            document_store: Document store with vector indices
            use_lightweight: Whether to use lightweight models
        """
        self.kb = kb
        self.document_store = document_store
        
        # Set up language model
        logger.info("Loading language model (this may take a moment)...")
        self._setup_llm(use_lightweight)
        
        # Set up RAG pipelines
        logger.info("Creating RAG query chains...")
        self._setup_rag_pipelines()
        
        # Free up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"Restaurant query engine ready with {len(document_store.restaurant_docs)} restaurant documents "
                  f"and {len(document_store.dish_docs)} dish documents")
    
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
        
        # Create the restaurant chain
        self.restaurant_chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in self.document_store.get_hybrid_restaurant_documents(x)]), 
             "question": RunnablePassthrough()}
            | restaurant_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Create the dish chain
        self.dish_chain = (
            {"context": lambda x: "\n\n".join([doc.page_content for doc in self.document_store.get_hybrid_dish_documents(x)]), 
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