import os
from datetime import datetime

# Base paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
VECTOR_CACHE_DIR = os.path.join(ROOT_DIR, "vector_cache")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

# Model configurations
EMBEDDING_MODELS = {
    "lightweight": "sentence-transformers/all-MiniLM-L6-v2",
    "standard": "sentence-transformers/all-mpnet-base-v2"
}

LLM_MODELS = {
    "lightweight": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "standard": "microsoft/phi-2"
}

# Helper functions
def get_latest_data_file(prefix="restaurants_", city=None):
    """Find the most recent data file for a given city or any city."""
    pattern = f"{prefix}"
    if city:
        pattern += f"{city}_"
    
    matching_files = []
    for file in os.listdir(ROOT_DIR):
        if file.startswith(pattern) and file.endswith(".json"):
            file_path = os.path.join(ROOT_DIR, file)
            matching_files.append((file_path, os.path.getmtime(file_path)))
    
    if not matching_files:
        return None
    
    # Return the most recent file
    return sorted(matching_files, key=lambda x: x[1], reverse=True)[0][0]