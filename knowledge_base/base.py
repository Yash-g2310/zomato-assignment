import json
import glob
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict

from .restaurant import Restaurant

class RestaurantKnowledgeBase:
    """Knowledge base for storing and querying restaurant information."""
    
    def __init__(self):
        self.restaurants = {}  # name -> Restaurant object
        self.cuisine_index = defaultdict(list)  # cuisine -> list of restaurant names
        self.location_words = defaultdict(list)  # location word -> list of restaurant names
        self.price_index = defaultdict(list)
        self.area_index = defaultdict(list)
    
    def add_restaurant(self, data: Dict[str, Any]) -> None:
        """Add a restaurant to the knowledge base."""
        restaurant = Restaurant(data)
        self.restaurants[restaurant.name] = restaurant
        
        # Index by cuisine
        if restaurant.cuisine and restaurant.cuisine != "Not found":
            cuisines = restaurant.cuisine.lower().split(", ")
            for cuisine in cuisines:
                self.cuisine_index[cuisine].append(restaurant.name)
        
        # Index by location words from address
        if restaurant.address and restaurant.address != "Not found":
            address_words = restaurant.address.lower().split()
            for word in address_words:
                if len(word) > 3:  # Skip very short words
                    self.location_words[word].append(restaurant.name)
        
        price_range = restaurant.get_price_range()
        price_category = "budget"
        if price_range["avg"] > 500:
            price_category = "premium"
        elif price_range["avg"] > 200:
            price_category = "moderate"
        self.price_index[price_category].append(restaurant.name)
        
        # Add geographic indexing
        if restaurant.address and restaurant.address != "Not found":
            area = self._extract_area_from_address(restaurant.address)
            if area:
                self.area_index[area].append(restaurant.name)
    
    def _extract_area_from_address(self, address: str) -> str:
        """Extract area name from address."""
        # Simple implementation - improve with regex patterns for your location format
        parts = address.split(',')
        if len(parts) >= 2:
            return parts[-2].strip().lower()
        return parts[0].strip().lower() if parts else ""
    
    def load_from_json(self, json_file: str) -> None:
        """Load restaurants from a JSON file produced by the scraper."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                restaurants_data = json.load(f)
                
            for restaurant_data in restaurants_data:
                self.add_restaurant(restaurant_data)
                
            print(f"Loaded {len(self.restaurants)} restaurants into knowledge base.")
            
        except Exception as e:
            print(f"Error loading restaurants from {json_file}: {str(e)}")
    
    def load_from_multiple_files(self, file_patterns: List[str]) -> None:
        """Load data from multiple JSON files matching the patterns."""
        loaded_count = 0
        for pattern in file_patterns:
            for json_file in glob.glob(pattern):
                print(f"Loading: {json_file}")
                self.load_from_json(json_file)
                loaded_count += 1
        
        print(f"Loaded data from {loaded_count} files")
    
    def get_restaurant(self, name: str) -> Optional[Restaurant]:
        """Get a restaurant by name."""
        return self.restaurants.get(name)
    
    def find_restaurants_by_name(self, query: str) -> List[Restaurant]:
        """Find restaurants with names containing the query."""
        query = query.lower()
        return [r for name, r in self.restaurants.items() if query in name.lower()]
    
    def find_restaurants_by_cuisine(self, cuisine: str) -> List[Restaurant]:
        """Find restaurants serving a specific cuisine."""
        cuisine = cuisine.lower()
        restaurant_names = set()
        
        # Check for exact matches
        if cuisine in self.cuisine_index:
            restaurant_names.update(self.cuisine_index[cuisine])
        
        # Check for partial matches
        for indexed_cuisine, names in self.cuisine_index.items():
            if cuisine in indexed_cuisine:
                restaurant_names.update(names)
        
        return [self.restaurants[name] for name in restaurant_names]
    
    def find_restaurants_by_location(self, location: str) -> List[Restaurant]:
        """Find restaurants in a specific location."""
        location = location.lower()
        restaurant_names = set()
        
        for word in location.split():
            if word in self.location_words:
                restaurant_names.update(self.location_words[word])
        
        return [self.restaurants[name] for name in restaurant_names]
    
    def find_restaurants_with_dish(self, dish_name: str) -> List[Dict[str, Any]]:
        """Find restaurants serving a specific dish."""
        dish_name = dish_name.lower()
        results = []
        
        for name, restaurant in self.restaurants.items():
            dishes = restaurant.find_dishes(dish_name)
            if dishes:
                results.append({
                    "restaurant_name": name,
                    "restaurant_info": restaurant.get_basic_info(),
                    "matching_dishes": dishes
                })
        
        return results
    
    def find_restaurants_by_price_range(self, price_category: str) -> List[Restaurant]:
        """Find restaurants in a specific price category (budget, moderate, premium)."""
        restaurant_names = self.price_index.get(price_category.lower(), [])
        return [self.restaurants[name] for name in restaurant_names]

    def find_restaurants_by_area(self, area: str) -> List[Restaurant]:
        """Find restaurants in a specific area."""
        area = area.lower()
        restaurant_names = set()
        
        # Check exact matches
        if area in self.area_index:
            restaurant_names.update(self.area_index[area])
        
        # Check partial matches
        for indexed_area, names in self.area_index.items():
            if area in indexed_area and indexed_area != area:
                restaurant_names.update(names)
                
        return [self.restaurants[name] for name in restaurant_names]
        
    def search_restaurants(self, query: Dict[str, Any]) -> List[Restaurant]:
        """Advanced search with multiple filters."""
        candidates = set(self.restaurants.keys())
        
        # Filter by cuisine if specified
        if query.get('cuisine'):
            cuisine_matches = set(name for cuisine in self.cuisine_index
                                 if query['cuisine'].lower() in cuisine.lower()
                                 for name in self.cuisine_index[cuisine])
            candidates &= cuisine_matches
        
        # Filter by area if specified
        if query.get('area'):
            area_matches = set()
            for area, names in self.area_index.items():
                if query['area'].lower() in area:
                    area_matches.update(names)
            candidates &= area_matches
        
        # Filter by price category if specified
        if query.get('price_category'):
            price_matches = set(self.price_index.get(query['price_category'].lower(), []))
            candidates &= price_matches
        
        # Filter by dish if specified
        if query.get('dish'):
            dish_results = self.find_restaurants_with_dish(query['dish'])
            dish_matches = {result['restaurant_name'] for result in dish_results}
            candidates &= dish_matches
        
        # Convert to Restaurant objects
        return [self.restaurants[name] for name in candidates]
        
    def get_restaurant_stats(self) -> Dict[str, Any]:
        """Get statistics about the restaurants in the knowledge base."""
        cuisines = {}
        for cuisine, restaurants in self.cuisine_index.items():
            cuisines[cuisine] = len(restaurants)
        
        return {
            "total_restaurants": len(self.restaurants),
            "cuisines": cuisines,
            "open_restaurants": sum(1 for r in self.restaurants.values() if r.status == "open"),
            "closed_restaurants": sum(1 for r in self.restaurants.values() if r.status == "closed"),
            "restaurants_with_ratings": sum(1 for r in self.restaurants.values() if r.ratings)
        }