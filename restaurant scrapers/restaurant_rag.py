import os
import json
import time
from typing import List, Dict, Any, Set
from tqdm import tqdm
import torch
import gc
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from knowledge_base import RestaurantKnowledgeBase
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RestaurantRAG:
    """RAG system for restaurant queries based on Zomato scraped data."""
    
    def __init__(self, json_file_path: str, use_lightweight: bool = True):
        """
        Initialize the RAG system with restaurant data.
        
        Args:
            json_file_path: Path to the Zomato JSON file
            use_lightweight: Use lightweight models to save memory
        """
        # Step 1: Load the knowledge base
        print("Loading restaurant knowledge base...")
        self.kb = RestaurantKnowledgeBase()
        self.kb.load_from_json(json_file_path)
        
        # Step 2: Create embeddings model - smaller for Roorkee dataset
        print("Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2" if use_lightweight else "sentence-transformers/all-mpnet-base-v2"
        )
        
        # Step 3: Create vector stores with caching
        self._setup_vector_stores(json_file_path)
        
        # Step 4: Initialize LLM with a lightweight model
        print("Loading language model (this may take a moment)...")
        self._setup_llm(use_lightweight)
        
        # Step 5: Set up RAG pipelines
        self._setup_rag_pipelines()
        
        print(f"Restaurant RAG ready with {len(self.restaurant_docs)} restaurant documents "
              f"and {len(self.dish_docs)} dish documents")
        
        # Force garbage collection to free memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def _setup_vector_stores(self, json_file_path: str):
        """Set up vector stores for restaurants and dishes with caching."""
        cache_dir = "vector_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache paths
        json_basename = os.path.basename(json_file_path).split('.')[0]
        restaurant_cache = os.path.join(cache_dir, f"{json_basename}_restaurants")
        dish_cache = os.path.join(cache_dir, f"{json_basename}_dishes")
        
        # Create or load restaurant vectors
        if os.path.exists(restaurant_cache):
            print("Loading cached restaurant embeddings...")
            self.restaurant_docs = []  # Still need to create for reference
            for name, restaurant in self.kb.restaurants.items():
                self.restaurant_docs.append(self._create_restaurant_document(name, restaurant))
            self.restaurant_vectorstore = FAISS.load_local(restaurant_cache, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating restaurant embeddings...")
            self.restaurant_docs = []
            for name, restaurant in tqdm(self.kb.restaurants.items(), desc="Processing restaurants"):
                self.restaurant_docs.append(self._create_restaurant_document(name, restaurant))
            
            self.restaurant_vectorstore = FAISS.from_documents(
                self.restaurant_docs, self.embeddings
            )
            self.restaurant_vectorstore.save_local(restaurant_cache)
        
        # Create or load dish vectors
        if os.path.exists(dish_cache):
            print("Loading cached dish embeddings...")
            self.dish_docs = []  # Still need to create for reference
            for restaurant_name, restaurant in self.kb.restaurants.items():
                self.dish_docs.extend(self._create_dish_documents_for_restaurant(restaurant_name, restaurant))
            self.dish_vectorstore = FAISS.load_local(dish_cache, self.embeddings, allow_dangerous_deserialization=True)
        else:
            print("Creating dish embeddings...")
            self.dish_docs = []
            # Count total dishes for progress bar
            total_dishes = sum(
                sum(1 for item in items if item.get('name'))
                for r in self.kb.restaurants.values()
                for category, items in r.menu_items.items()
            )
            
            with tqdm(total=total_dishes, desc="Processing dishes") as pbar:
                for restaurant_name, restaurant in self.kb.restaurants.items():
                    new_docs = self._create_dish_documents_for_restaurant(restaurant_name, restaurant, pbar)
                    self.dish_docs.extend(new_docs)
            
            self.dish_vectorstore = FAISS.from_documents(
                self.dish_docs, self.embeddings
            )
            self.dish_vectorstore.save_local(dish_cache)
    
    def _create_restaurant_document(self, name: str, restaurant) -> Document:
        """Create a document for a restaurant."""
        info = restaurant.get_basic_info()
        price_range = restaurant.get_price_range()
        
        # Create structured content
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
        
        return Document(
            page_content=content,
            metadata={
                "name": name,
                "cuisine": info['cuisine'],
                "status": info['status'],
                "address": info['address'],
                "min_price": price_range['min'],
                "max_price": price_range['max'],
                "avg_price": price_range['avg']
            }
        )
    
    def _create_dish_documents_for_restaurant(self, restaurant_name: str, restaurant, pbar=None) -> List[Document]:
        """Create document objects for menu items in a restaurant."""
        documents = []
        seen_dishes: Set[str] = set()  # Track unique dishes to avoid duplicates
        
        for category, items in restaurant.menu_items.items():
            for item in items:
                name = item.get('name', '')
                if not name:
                    continue
                
                # Create a unique identifier for the dish
                dish_id = f"{name}_{item.get('price', '')}"
                
                # Skip duplicates
                if dish_id in seen_dishes:
                    if pbar:
                        pbar.update(1)
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
                            "type": item.get('type', 'unknown')
                        }
                    )
                )
                
                if pbar:
                    pbar.update(1)
        
        return documents
    
    def _setup_llm(self, use_lightweight: bool = True):
        """Set up a Hugging Face language model that works on limited hardware."""
        try:
            # If we want lightweight model or have limited RAM
            if use_lightweight:
                model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                print(f"Using lightweight model: {model_id}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto",
                    load_in_4bit=True,
                )
            else:
                # Try to use more powerful model
                model_id = "microsoft/phi-2"
                print(f"Using standard model: {model_id}")
                
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                )
                
            # Create a text generation pipeline
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
            print(f"Error loading model: {str(e)}")
            print("Falling back to smallest available model...")
            
            # Fallback to the smallest model possible
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                load_in_4bit=True,
            )
            
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
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
    
    def _setup_rag_pipelines(self):
        """Set up the RAG pipelines for different query types."""
        # Restaurant query pipeline - updated for Roorkee restaurants
        restaurant_template = """You are a helpful restaurant assistant with knowledge about restaurants in Roorkee, India.
Answer the user's question based only on the following restaurant information:

{context}

User question: {question}

Answer in a helpful, concise way using only the information provided above.
If multiple restaurants match the query, list 2-3 options with their key details.
If the information doesn't contain the answer, say "I don't have that information."
"""
        restaurant_prompt = ChatPromptTemplate.from_template(restaurant_template)
        
        self.restaurant_rag_chain = (
            {"context": self.restaurant_vectorstore.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
            | restaurant_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Dish query pipeline - updated for Indian dishes in Roorkee
        dish_template = """You are a helpful restaurant menu specialist with knowledge about dishes served in Roorkee, India.
Answer the user's question based only on the following dish information:

{context}

User question: {question}

Answer in a helpful, concise way using only the information provided above.
If many dishes match, focus on the most relevant ones and include their prices.
If the information doesn't contain the answer, say "I don't have that information."
"""
        dish_prompt = ChatPromptTemplate.from_template(dish_template)
        
        self.dish_rag_chain = (
            {"context": self.dish_vectorstore.as_retriever(search_kwargs={"k": 7}), "question": RunnablePassthrough()}
            | dish_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str) -> str:
        """
        Query the RAG system with a user question.
        The system will determine if the question is about restaurants or dishes.
        """
        start_time = time.time()
        
        # Improved heuristic for Roorkee dataset
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
        
        # Determine query type and process accordingly
        if has_price_keywords or (dish_score > 0 and restaurant_score > 0):
            # Questions about prices or complex queries should use both
            print("Processing as combined query (restaurant + dishes)...")
            rest_answer = self.restaurant_rag_chain.invoke(question)
            dish_answer = self.dish_rag_chain.invoke(question)
            
            if "I don't have that information" in rest_answer and "I don't have that information" in dish_answer:
                answer = "I don't have that specific information in my knowledge base."
            elif "I don't have that information" in rest_answer:
                answer = dish_answer
            elif "I don't have that information" in dish_answer:
                answer = rest_answer
            else:
                # Combine the answers in a meaningful way
                answer = f"{rest_answer}\n\nAdditional dish information: {dish_answer}"
        elif dish_score > restaurant_score:
            print("Processing as dish query...")
            answer = self.dish_rag_chain.invoke(question)
        else:
            print("Processing as restaurant query...")
            answer = self.restaurant_rag_chain.invoke(question)
            
        # Clean up the response - remove model-specific formatting
        answer = answer.replace("<|im_end|>", "").replace("<|im_start|>assistant", "").strip()
        
        # Clean up any trailing text that might come from the model
        if "<" in answer and ">" in answer:
            # Try to cut at the first tag after the main content
            main_content = answer.split("<")[0].strip()
            if main_content:
                answer = main_content
        
        elapsed_time = time.time() - start_time
        print(f"Query processed in {elapsed_time:.2f} seconds")
        
        return answer


def find_latest_json_file():
    """Find the latest Zomato JSON file in the current directory."""
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(parent_dir)  # Change to script's directory
    
    # Look in current directory and parent directory
    json_files = [f for f in os.listdir('.') if f.startswith('zomato_') and f.endswith('.json')]
    if not json_files and os.path.exists('..'):
        # Try parent directory
        parent_json_files = [os.path.join('..', f) for f in os.listdir('..') 
                            if f.startswith('zomato_') and f.endswith('.json')]
        json_files.extend(parent_json_files)
        
    if not json_files:
        raise FileNotFoundError("No Zomato JSON files found. Run the scraper first.")
    
    # Return the latest file by modification time
    return max(json_files, key=lambda f: os.path.getmtime(f))


def demo_rag():
    """Run a demo of the Restaurant RAG system."""
    # Find the latest JSON file
    try:
        json_file = find_latest_json_file()
        print(f"Using data from: {json_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Running the knowledge base to create the data...")
        
        # Import and run the knowledge base
        from knowledge_base import main as kb_main
        kb_main()
        
        # Try finding the file again
        json_file = find_latest_json_file()
        print(f"Created and using data from: {json_file}")
    
    # Check if we should use lightweight mode
    use_lightweight = input("Use lightweight model to save memory? (y/n, default: y): ").lower() != 'n'
    
    # Create the RAG system
    try:
        rag = RestaurantRAG(json_file, use_lightweight=use_lightweight)
    except Exception as e:
        print(f"Error initializing RAG with standard settings: {e}")
        print("Falling back to lightweight mode...")
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
            answer = rag.query(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
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
            answer = rag.query(user_question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    demo_rag()