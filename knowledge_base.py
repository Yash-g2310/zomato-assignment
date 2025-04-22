import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
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


class RestaurantKnowledgeBase:
    """Knowledge base for storing and querying restaurant information."""
    
    def __init__(self):
        self.restaurants = {}  # name -> Restaurant object
        self.cuisine_index = defaultdict(list)  # cuisine -> list of restaurant names
        self.location_words = defaultdict(list)  # location word -> list of restaurant names
    
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
    
    def load_from_json(self, json_file: str) -> None:
        """Load restaurants from a JSON file produced by the Zomato scraper."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                restaurants_data = json.load(f)
                
            for restaurant_data in restaurants_data:
                self.add_restaurant(restaurant_data)
                
            print(f"Loaded {len(self.restaurants)} restaurants into knowledge base.")
            
        except Exception as e:
            print(f"Error loading restaurants from {json_file}: {str(e)}")
    
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


def run_scraper_if_needed(city_url: str = "https://www.zomato.com/roorkee") -> str:
    """
    Run the Zomato scraper if needed and return the path to the JSON output file.
    
    Args:
        city_url: URL of the city page on Zomato
        
    Returns:
        Path to the JSON output file
    """
    # Check if we already have recent data
    city_name = city_url.split('/')[-1]
    today = datetime.now().strftime('%Y%m%d')
    expected_file = f"zomato_{city_name}_{today}.json"
    
    if os.path.exists(expected_file):
        print(f"Using existing data file: {expected_file}")
        return expected_file
    
    # Run the scraper
    print(f"Running Zomato scraper for {city_url}...")
    try:
        # First try importing and running directly
        try:
            from zomato_scrapper import ZomatoScraper
            scraper = ZomatoScraper(headless=True, cache=True)
            restaurants = scraper.scrape_all_restaurants(city_url)
            
            # Save to JSON if not already saved by the scraper
            if not os.path.exists(expected_file):
                with open(expected_file, 'w', encoding='utf-8') as f:
                    json.dump(restaurants, f, indent=2, ensure_ascii=False)
            
            print(f"Saved data to {expected_file}")
            
        except ImportError:
            # Fall back to running as a separate process
            subprocess.run([sys.executable, "zomato_scrapper.py"], check=True)
            
            # Check if the file was created
            if not os.path.exists(expected_file):
                # Look for any zomato_ JSON files
                json_files = [f for f in os.listdir('.') if f.startswith('zomato_') and f.endswith('.json')]
                if json_files:
                    expected_file = json_files[0]
                else:
                    raise FileNotFoundError(f"Could not find output from Zomato scraper")
        
        return expected_file
        
    except Exception as e:
        print(f"Error running Zomato scraper: {str(e)}")
        # Look for any existing zomato_ JSON files as fallback
        json_files = [f for f in os.listdir('.') if f.startswith('zomato_') and f.endswith('.json')]
        if json_files:
            print(f"Using existing data file: {json_files[0]}")
            return json_files[0]
        raise


def main():
    """Main function to run the knowledge base creation."""
    # Get the city URL from command line arguments or use default
    city_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.zomato.com/roorkee"
    
    # Run the scraper if needed
    try:
        json_file = run_scraper_if_needed(city_url)
    except Exception as e:
        print(f"Failed to get restaurant data: {str(e)}")
        return
    
    # Create and load the knowledge base
    kb = RestaurantKnowledgeBase()
    kb.load_from_json(json_file)
    
    # Show some basic stats
    stats = kb.get_restaurant_stats()
    print(f"\n--- Knowledge Base Statistics ---")
    print(f"Total restaurants: {stats['total_restaurants']}")
    print(f"Open restaurants: {stats['open_restaurants']}")
    print(f"Closed restaurants: {stats['closed_restaurants']}")
    print(f"Top cuisines:")
    top_cuisines = sorted(stats['cuisines'].items(), key=lambda x: x[1], reverse=True)[:5]
    for cuisine, count in top_cuisines:
        print(f"  - {cuisine.title()}: {count} restaurants")
    
    # Example queries to demonstrate functionality
    print("\n--- Example Queries ---")
    
    # Find restaurants by cuisine
    if stats['cuisines']:
        example_cuisine = top_cuisines[0][0]
        cuisine_restaurants = kb.find_restaurants_by_cuisine(example_cuisine)
        print(f"\nRestaurants serving {example_cuisine.title()} cuisine ({len(cuisine_restaurants)}):")
        for i, restaurant in enumerate(cuisine_restaurants[:3], 1):
            print(f"  {i}. {restaurant.name} - {restaurant.address}")
        if len(cuisine_restaurants) > 3:
            print(f"  ...and {len(cuisine_restaurants) - 3} more")
    
    # Find restaurants with a common dish
    example_dishes = ["paneer", "chicken", "biryani", "pizza", "burger"]
    for dish in example_dishes:
        restaurants_with_dish = kb.find_restaurants_with_dish(dish)
        if restaurants_with_dish:
            print(f"\nRestaurants serving {dish} ({len(restaurants_with_dish)}):")
            for i, result in enumerate(restaurants_with_dish[:3], 1):
                print(f"  {i}. {result['restaurant_name']} - {len(result['matching_dishes'])} matching dishes")
                for j, dish_detail in enumerate(result['matching_dishes'][:2], 1):
                    dish_name = dish_detail.get('name', 'Unknown')
                    dish_price = dish_detail.get('price', 'Price not available')
                    print(f"     {j}. {dish_name} - {dish_price}")
                if len(result['matching_dishes']) > 2:
                    print(f"     ...and {len(result['matching_dishes']) - 2} more")
            if len(restaurants_with_dish) > 3:
                print(f"  ...and {len(restaurants_with_dish) - 3} more restaurants")
            break
    
    print("\nKnowledge base is ready for queries!")


if __name__ == "__main__":
    main()