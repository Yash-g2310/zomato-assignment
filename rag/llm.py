import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

def create_llm(use_lightweight=False, temperature=0.2):
    """
    Create an LLM model with configurable size and parameters.
    """
    # First try local models (no API token needed)
    try:
        logger.info("Trying to create local LLM...")
        from langchain_community.llms import HuggingFacePipeline
        
        # Choose an appropriate local model based on size preference
        if use_lightweight:
            model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        else:
            model_id = "microsoft/phi-2"  # Relatively small but powerful
            
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=temperature
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        logger.info(f"Local LLM ready: {model_id}")
        return llm
            
    except Exception as e:
        logger.warning(f"Could not create local LLM: {e}")
        
    # Try API-based approach if local approach failed
    try:
        # Only prompt for token if environment variable isn't set
        hf_token = os.environ.get("HUGGINGFACE_API_KEY")
        
        if not hf_token:
            logger.warning("No Hugging Face API token found in environment.")
            logger.warning("For better performance, set the HUGGINGFACE_API_KEY environment variable.")
            logger.warning("Falling back to fake LLM for testing.")
            # Skip prompting for API token and use fake LLM instead
            raise ValueError("No API token available")

        # Only proceed with API if token is available in environment
        from langchain_huggingface import HuggingFaceEndpoint
        
        # Choose appropriate model based on size preference
        if use_lightweight:
            logger.info("Creating lightweight API-based LLM...")
            model_id = "google/flan-t5-base"  # Much smaller model that works well with the API
        else:
            logger.info("Creating full capability API-based LLM...")
            model_id = "google/flan-t5-xl"  # Larger but still API-friendly
        
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            huggingfacehub_api_token=hf_token,
            task="text2text-generation",  # Specify correct task
            model_kwargs={"temperature": temperature, "max_length": 512}
        )
        
        logger.info(f"API-based LLM ready: {model_id}")
        return llm
    
    except Exception as e:
        logger.error(f"Error creating API-based LLM: {e}")
        
        # Last resort - use a fake LLM for testing
        from langchain.llms.fake import FakeListLLM
        
        logger.warning("Using MOCK LLM - for testing only!")
        return FakeListLLM(responses=[
            "This is a test response about restaurants in Roorkee. There are several restaurants including Sagar Ratna, Dominoz, and McDonald's offering various cuisines.",
            "There are multiple restaurants with vegetarian options in Roorkee. Some popular ones include Sagar Ratna, Café Coffee Day, and local eateries that serve traditional vegetarian dishes.",
            "Based on the menu information, several restaurants serve butter chicken including Punjabi Tadka, Royal Kitchen, and others in the Roorkee area.",
            "Pizza prices in Roorkee restaurants typically range from ₹200 to ₹600 depending on size and toppings. Domino's and Pizza Hut offer options in this price range.",
            "Yes, there are several bakeries in Roorkee. Some of them are open until late evening around 9-10 PM."
        ])