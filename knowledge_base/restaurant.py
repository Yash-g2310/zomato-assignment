import json
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

class Restaurant:
    """Restaurant knowledge base entry with all relevant information."""
    
    def __init__(self, data: Dict[str, Any]):
        self.name = data.get("name", "Unknown Restaurant")
        self.url = data.get("url", "")
        self.cuisine = data.get("cuisine", "Not specified")
        self.address = data.get("address", "Not available")
        self.phone = data.get("phone", "Not available")
        self.status = data.get("status", "unknown")
        self.ratings = data.get("ratings", {})
        self.menu_items = data.get("menu_items", {})
        self.scraped_at = data.get("scraped_at", "")
        self.source = data.get("source", "unknown")
        self.sources = [self.source]
        self.last_updated = datetime.now().isoformat()
        
        # Process and index the menu for better search
        self.menu_index = self._index_menu()
        self.dish_categories = list(self.menu_items.keys())
        self.dish_count = sum(len(items) for items in self.menu_items.values())
        self.has_veg_options = any(item.get("type") == "veg" 
                                 for items in self.menu_items.values() 
                                 for item in items)
    
    def _index_menu(self) -> Dict[str, List[Dict]]:
        """Create searchable index of menu items."""
        menu_index = defaultdict(list)
        for category, items in self.menu_items.items():
            for item in items:
                name = item.get("name", "").lower()
                if name:
                    # Store item with its category
                    item_with_category = {**item, "category": category}
                    # Use append instead of direct assignment
                    menu_index[name].append(item_with_category)

                    # Also index by keywords from name
                    for word in name.split():
                        if len(word) > 2:  # Skip very short words
                            menu_index[word].append(item_with_category)

        return dict(menu_index)
    
    def get_basic_info(self) -> Dict[str, Any]:
        """Return basic information about the restaurant."""
        return {
            "name": self.name,
            "cuisine": self.cuisine,
            "address": self.address,
            "phone": self.phone,
            "status": self.status,
            "ratings": self.ratings,
            "dish_count": self.dish_count,
            "has_veg_options": self.has_veg_options
        }
    
    def find_dishes(self, query: str) -> List[Dict[str, Any]]:
        """Search for dishes containing the query in their name."""
        query = query.lower()
        results = []
        
        # Check exact matches first
        if query in self.menu_index:
            return self.menu_index[query]  # Now always returns a list
        
        # Then partial matches
        for dish_name, items in self.menu_index.items():
            if query in dish_name and dish_name != query:  # Avoid duplicate results
                results.extend(items)
        
        return results
    
    def get_dishes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get all dishes in a specified category."""
        return self.menu_items.get(category, [])
    
    def get_dishes_by_type(self, veg_type: str) -> List[Dict[str, Any]]:
        """Get all dishes of a specific type (veg/non-veg)."""
        results = []
        for category, items in self.menu_items.items():
            for item in items:
                if item.get("type") == veg_type:
                    results.append({**item, "category": category})
        
        return results
    
    def get_price_range(self) -> Dict[str, float]:
        """Get the minimum and maximum price of items on the menu."""
        prices = []
        for category, items in self.menu_items.items():
            for item in items:
                price_text = item.get("price", "")
                if price_text and price_text != "Price not available":
                    # Extract numeric value from price text
                    price_value = ''.join(filter(lambda x: x.isdigit() or x == '.', price_text))
                    try:
                        prices.append(float(price_value))
                    except ValueError:
                        continue
        
        if not prices:
            return {"min": 0, "max": 0, "avg": 0}
        
        return {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices)
        }