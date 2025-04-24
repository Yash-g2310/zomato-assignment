import os
import json
from datetime import datetime
import argparse
from typing import List, Dict, Any
import logging
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import scrapers
from scrapers.zomato_scrapper import ZomatoScraper
from scrapers.swiggy_scrapper import SwiggyScraper

DEFAULT_CITIES = [
    "roorkee",
    "ncr",
    "delhi",
]

def standardize_restaurant_data(data: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Standardize restaurant data format between Zomato and Swiggy."""
    
    # Ensure all necessary fields exist
    standardized = {
        "name": data.get("name", "Unknown Restaurant"),
        "url": data.get("url", ""),
        "source": source,
        "scraped_at": data.get("scraped_at", datetime.now().isoformat()),
    }
    
    # Handle cuisine differences
    standardized["cuisine"] = data.get("cuisine", "Not specified")
    
    # Handle address field
    standardized["address"] = data.get("address", "Not available")
    
    # Handle status field (Swiggy doesn't have this)
    if source == "zomato":
        standardized["status"] = data.get("status", "unknown")
    else:  # source == "swiggy"
        # Assume open unless specifically marked closed
        standardized["status"] = "open"
    
    # Handle ratings with consistent format
    if source == "zomato":
        standardized["ratings"] = data.get("ratings", {})
    else:  # source == "swiggy"
        rating = data.get("rating")
        if rating:
            standardized["ratings"] = {"Dining": rating}
        else:
            standardized["ratings"] = {}
    
    # Handle phone field
    standardized["phone"] = data.get("phone", "Not available")
    
    # Handle menu items and enhance if needed
    menu_items = data.get("menu_items", {})
    
    # For Swiggy, add empty description and unknown type to match Zomato structure
    if source == "swiggy":
        for category, items in menu_items.items():
            for item in items:
                if "description" not in item:
                    item["description"] = ""
                if "type" not in item:
                    item["type"] = "unknown"
    
    standardized["menu_items"] = menu_items
    
    # Handle order link
    standardized["order_link"] = data.get("order_link", data.get("url", ""))
    
    return standardized

def deduplicate_restaurants(all_restaurants):
    """Smarter restaurant deduplication handling slight name variations."""
    unique_restaurants = []
    seen_signatures = set()
    
    for restaurant in all_restaurants:
        # Create a normalized signature for comparison
        name = restaurant["name"].lower().strip()
        
        # Remove common suffixes and prefixes for better matching
        normalized_name = re.sub(r'^(the|hotel|restaurant)\s+', '', name)
        normalized_name = re.sub(r'\s+(restaurant|hotel|cafe)$', '', normalized_name)
        
        # Remove spaces and punctuation for fuzzy matching
        simple_name = ''.join(c for c in normalized_name if c.isalnum())
        
        # Add location info if available to differentiate branches
        address = restaurant.get("address", "").lower()
        location_words = []
        if address and address != "not available":
            # Extract likely location identifiers from address
            location_words = [word for word in address.split() 
                            if len(word) > 3 and word not in ['road', 'street', 'near']]
        
        # Create signature with location if available to differentiate branches
        if location_words:
            signature = f"{simple_name}_{location_words[0]}"
        else:
            signature = simple_name
        
        # Skip if we've seen this signature
        if signature in seen_signatures:
            logger.info(f"Skipping potential duplicate: {restaurant['name']}")
            continue
        
        seen_signatures.add(signature)
        unique_restaurants.append(restaurant)
    
    logger.info(f"Removed {len(all_restaurants) - len(unique_restaurants)} duplicates")
    return unique_restaurants

def run_combined_scraper(city_name: str, force: bool = False) -> str:
    """Run both Zomato and Swiggy scrapers and combine the results."""
    logger.info(f"Starting combined scraper for {city_name}")
    
    today = datetime.now().strftime('%Y%m%d')
    output_file = f"restaurants_{city_name}_{today}.json"
    
    # Check if we already have today's data
    if os.path.exists(output_file) and not force:
        logger.info(f"Using existing combined data file: {output_file}")
        return output_file
    
    # Initialize empty lists for resilience
    zomato_restaurants = []
    swiggy_restaurants = []
    
    try:
        # Run Zomato scraper - URL format: zomato.com/{city_name}
        logger.info("Running Zomato scraper...")
        zomato_scraper = ZomatoScraper(headless=True, cache=True)
        zomato_url = f"https://www.zomato.com/{city_name}"
        
        try:
            zomato_restaurants = zomato_scraper.scrape_all_restaurants(zomato_url)
            logger.info(f"Found {len(zomato_restaurants)} restaurants from Zomato")
        except Exception as e:
            logger.error(f"Zomato scraping failed: {e}")
        
        # Run Swiggy scraper - URL format: swiggy.com/city/{city_name}
        logger.info("Running Swiggy scraper...")
        swiggy_scraper = SwiggyScraper(headless=True, cache=True)
        swiggy_url = f"https://www.swiggy.com/city/{city_name}"
        
        try:
            swiggy_restaurants = swiggy_scraper.scrape_all_restaurants(swiggy_url)
            logger.info(f"Found {len(swiggy_restaurants)} restaurants from Swiggy")
        except Exception as e:
            logger.error(f"Swiggy scraping failed: {e}")
    
    except Exception as e:
        logger.error(f"Fatal error in combined scraper: {e}")
        # Try to find existing data as fallback
        existing_files = [f for f in os.listdir('.') if f.startswith(f"restaurants_{city_name}_")]
        if existing_files:
            latest_file = sorted(existing_files)[-1]
            logger.info(f"Using existing data file as fallback: {latest_file}")
            return latest_file
    
    # Standardize and combine data
    all_restaurants = []
    
    # Standardize Zomato data
    for restaurant in zomato_restaurants:
        standardized = standardize_restaurant_data(restaurant, "zomato")
        all_restaurants.append(standardized)
    
    # Standardize Swiggy data
    for restaurant in swiggy_restaurants:
        standardized = standardize_restaurant_data(restaurant, "swiggy")
        all_restaurants.append(standardized)
    
    # Skip deduplication if no restaurants from one source
    if zomato_restaurants and swiggy_restaurants:
        # Deduplicate restaurants
        all_restaurants = deduplicate_restaurants(all_restaurants)
    
    # Save combined data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_restaurants, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(all_restaurants)} restaurants to {output_file}")
    return output_file

def scrape_multiple_cities(cities: List[str] = None, force: bool = False, delay: int = 5) -> List[str]:
    """
    Scrape multiple cities in sequence.
    
    Args:
        cities: List of cities to scrape. If None, uses DEFAULT_CITIES.
        force: Whether to force re-scraping even if data exists
        delay: Delay in seconds between cities to avoid overloading servers
        
    Returns:
        List of output file paths
    """
    if cities is None:
        cities = DEFAULT_CITIES
    
    results = []
    
    for i, city in enumerate(cities):
        logger.info(f"Processing city {i+1}/{len(cities)}: {city}")
        
        try:
            output_file = run_combined_scraper(city, force)
            results.append(output_file)
            
            # Add delay between cities except for the last one
            if i < len(cities) - 1:
                logger.info(f"Sleeping for {delay} seconds before next city...")
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Failed to scrape city {city}: {e}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run combined Zomato and Swiggy scraping")
    parser.add_argument("--city", help="Single city name to scrape (e.g., roorkee, ncr)")
    parser.add_argument("--cities", nargs="+", help="List of cities to scrape")
    parser.add_argument("--force", action="store_true", help="Force scraping even if data exists")
    parser.add_argument("--delay", type=int, default=5, help="Delay between cities in seconds (default: 5)")
    args = parser.parse_args()
    
    if args.city:
        # Single city mode
        output_file = run_combined_scraper(args.city, args.force)
        print(f"Combined data saved to: {output_file}")
    else:
        # Multiple cities mode
        cities_to_process = args.cities if args.cities else DEFAULT_CITIES
        output_files = scrape_multiple_cities(cities_to_process, args.force, args.delay)
        
        print(f"\nProcessed {len(output_files)} cities:")
        for city_file in output_files:
            print(f"  - {city_file}")