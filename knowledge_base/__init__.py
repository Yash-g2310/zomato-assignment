"""
Restaurant Knowledge Base Package

This package provides classes and utilities for working with restaurant data.
"""

from .restaurant import Restaurant
from .base import RestaurantKnowledgeBase
from .scraper_utils import run_scraper_if_needed, run_combined_scraper
from .kb_manager import KnowledgeBaseManager

# For backward compatibility
import sys
import os

# Expose main function similar to the original knowledge_base.py
def main():
    """Main function to run the knowledge base creation."""
    from .scraper_utils import run_scraper_if_needed
    import sys
    
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