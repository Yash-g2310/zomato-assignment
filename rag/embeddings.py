import logging
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class RestaurantEmbeddings:
    """Embedding models for restaurant data."""
    
    @staticmethod
    def create_embeddings(use_lightweight: bool = True):
        """
        Create an embeddings model suitable for restaurant data.
        
        Args:
            use_lightweight: If True, use a smaller, faster model
            
        Returns:
            HuggingFaceEmbeddings model
        """
        logger.info(f"Creating {'lightweight' if use_lightweight else 'standard'} embedding model...")
        
        model_name = "sentence-transformers/all-MiniLM-L6-v2" if use_lightweight else "sentence-transformers/all-mpnet-base-v2"
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        logger.info("Embedding model ready")
        return embeddings