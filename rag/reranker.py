import logging
from typing import List, Dict, Any
from langchain.schema import Document

logger = logging.getLogger(__name__)

class QueryDocumentReranker:
    """Reranks retrieved documents based on relevance to the query."""
    
    def __init__(self, use_lightweight=True):
        self.model = self._load_reranker_model(use_lightweight)
        
    def _load_reranker_model(self, use_lightweight=True):
        """Load a cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            
            if use_lightweight:
                model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            else:
                model_name = "cross-encoder/ms-marco-MiniLM-L-12-v2"
                
            logger.info(f"Loading reranker model: {model_name}")
            model = CrossEncoder(model_name)
            logger.info("Reranker model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Error loading reranker: {e}")
            logger.warning("Continuing without reranking capability")
            return None
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The user query
            documents: List of retrieved documents
            top_k: Number of top documents to return
            
        Returns:
            Reranked list of documents
        """
        if not self.model or not documents:
            return documents[:top_k]
            
        try:
            # Prepare sentence pairs for the cross-encoder
            sentence_pairs = [(query, doc.page_content) for doc in documents]
            
            # Get scores from the cross-encoder
            scores = self.model.predict(sentence_pairs)
            
            # Create (score, doc) pairs and sort by score in descending order
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return the top-k documents
            return [doc for _, doc in scored_docs[:top_k]]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:top_k]