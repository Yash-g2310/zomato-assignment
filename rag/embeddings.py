import logging
from typing import Optional

logger = logging.getLogger(__name__)

def create_embedding_model(use_lightweight=False, device: str = None):
    """
    Create an embedding model with configurable size and device.
    
    Args:
        use_lightweight: Whether to use a lightweight model for faster performance
        device: Device to use (cpu, cuda, mps)
    """
    try:
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings
        
        # Model selection based on performance vs speed tradeoff
        if use_lightweight:
            logger.info("Creating lightweight embedding model...")
            model_name = "BAAI/bge-small-en-v1.5"
            model_kwargs = {'device': device or 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
        else:
            logger.info("Creating full-size embedding model...")
            model_name = "BAAI/bge-large-en-v1.5"  # Much better semantic understanding
            model_kwargs = {'device': device or 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info("Embedding model ready")
        return embeddings
        
    except Exception as e:
        logger.error(f"Error creating embedding model: {e}")
        # Fall back to older API if needed
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        if use_lightweight:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            model_name = "sentence-transformers/all-mpnet-base-v2"
            
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info("Embedding model ready (fallback)")
        return embeddings