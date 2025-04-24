import os
import json
import pickle
import logging
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional

from .base import RestaurantKnowledgeBase

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """Manager for restaurant knowledge base persistence and updates."""
    
    def __init__(self, cache_dir: str = "kb_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.kb = RestaurantKnowledgeBase()
    
    def save_kb(self, filename: str = None) -> str:
        """Save the knowledge base to a pickle file."""
        if filename is None:
            filename = f"kb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        filepath = os.path.join(self.cache_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.kb, f)
        
        logger.info(f"Knowledge base saved to {filepath}")
        return filepath
    
    def load_kb(self, filename: str = None) -> bool:
        """Load the knowledge base from a pickle file."""
        if filename is None:
            # Find the latest KB file
            kb_files = [f for f in os.listdir(self.cache_dir) if f.startswith("kb_") and f.endswith(".pkl")]
            if not kb_files:
                logger.warning("No knowledge base files found.")
                return False
                
            filename = sorted(kb_files)[-1]  # Get the latest file
        
        filepath = os.path.join(self.cache_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                self.kb = pickle.load(f)
            
            logger.info(f"Knowledge base loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {str(e)}")
            return False
    
    def update_from_json_files(self, file_patterns: List[str]) -> int:
        """Update the knowledge base with data from JSON files."""
        prev_count = len(self.kb.restaurants)

        # Add debugging to count total restaurants in files
        total_in_files = 0
        restaurant_names = set()
        duplicate_count = 0

        for pattern in file_patterns:
            for json_file in glob.glob(pattern):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    logger.info(f"File {json_file} contains {len(data)} restaurants")
                    total_in_files += len(data)

                    # Count duplicates
                    for restaurant in data:
                        name = restaurant.get("name", "Unknown")
                        if name in restaurant_names:
                            duplicate_count += 1
                        restaurant_names.add(name)
                except:
                    logger.error(f"Error analyzing {json_file}")

        logger.info(f"Total restaurants in files: {total_in_files}")
        logger.info(f"Unique restaurant names: {len(restaurant_names)}")
        logger.info(f"Duplicate restaurant names: {duplicate_count}")

        self.kb.load_from_multiple_files(file_patterns)
        new_count = len(self.kb.restaurants)

        logger.info(f"Added {new_count - prev_count} restaurants to knowledge base")
        return new_count - prev_count
    
    def export_to_json(self, output_file: str = None) -> str:
        """Export the knowledge base to a JSON file."""
        if output_file is None:
            output_file = f"exported_kb_{datetime.now().strftime('%Y%m%d')}.json"
        
        data = []
        for name, restaurant in self.kb.restaurants.items():
            restaurant_data = {
                "name": restaurant.name,
                "url": restaurant.url,
                "cuisine": restaurant.cuisine,
                "address": restaurant.address,
                "phone": restaurant.phone,
                "status": restaurant.status,
                "ratings": restaurant.ratings,
                "menu_items": restaurant.menu_items,
                "source": restaurant.source,
                "scraped_at": restaurant.scraped_at,
                "last_updated": restaurant.last_updated
            }
            data.append(restaurant_data)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported {len(data)} restaurants to {output_file}")
        return output_file
    
    def get_kb_stats(self) -> Dict[str, Any]:
        """Get statistics for the knowledge base."""
        return self.kb.get_restaurant_stats()