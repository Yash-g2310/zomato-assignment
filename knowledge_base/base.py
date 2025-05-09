import json
import glob
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


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

        # Create a city-specific key by extracting city from address or using source file
        city = "unknown"
        if restaurant.address:
            address_parts = restaurant.address.split(',')
            if len(address_parts) >= 2:
                city = address_parts[-1].strip()

        # Create unique key combining name and city
        unique_key = f"{restaurant.name} ({city})"
        self.restaurants[unique_key] = restaurant
                
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
    
    def load_from_json(self, json_file: str) -> int:
        """Load restaurant data from a JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate data format
            if not isinstance(data, list):
                raise ValueError(f"Expected a list of restaurants, got {type(data)}")

            if len(data) == 0:
                logger.warning(f"The file {json_file} contains an empty restaurant list!")
                return 0

            # Sample check for required fields in the first restaurant
            if data and isinstance(data[0], dict):
                required_fields = ['name', 'menu_items']
                missing = [field for field in required_fields if field not in data[0]]
                if missing:
                    logger.error(f"Missing required fields in restaurant data: {missing}")
                    raise ValueError(f"JSON format invalid - missing fields: {missing}")

            # Process restaurants
            count = 0
            for restaurant_data in data:
                try:
                    self.add_restaurant(restaurant_data)
                    count += 1
                except Exception as e:
                    logger.error(f"Error adding restaurant {restaurant_data.get('name', 'unknown')}: {str(e)}")

            logger.info(f"Loaded {count} restaurants into knowledge base.")
            return count

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {json_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {json_file}: {str(e)}")
            raise
    
    def load_from_multiple_files(self, file_patterns: List[str]) -> int:
        """Load data from multiple JSON files matching the patterns."""
        loaded_count = 0
        total_restaurants = 0
        for pattern in file_patterns:
            for json_file in glob.glob(pattern):
                logger.info(f"Loading: {json_file}")
                restaurants = self.load_from_json(json_file)
                total_restaurants += restaurants
                loaded_count += 1
    
        logger.info(f"Loaded {total_restaurants} restaurants from {loaded_count} files")
        return total_restaurants
    
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