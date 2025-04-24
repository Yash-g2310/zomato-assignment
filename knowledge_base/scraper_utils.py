import os
import sys
import json
import subprocess
from datetime import datetime

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
            from scrapers.zomato_scrapper import ZomatoScraper
            scraper = ZomatoScraper(headless=True, cache=True)
            restaurants = scraper.scrape_all_restaurants(city_url)
            
            # Save to JSON if not already saved by the scraper
            if not os.path.exists(expected_file):
                with open(expected_file, 'w', encoding='utf-8') as f:
                    json.dump(restaurants, f, indent=2, ensure_ascii=False)
            
            print(f"Saved data to {expected_file}")
            
        except ImportError:
            # Fall back to running as a separate process
            subprocess.run([sys.executable, "scrapers/zomato_scrapper.py"], check=True)
            
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

def run_combined_scraper(city: str) -> str:
    """Run the combined scraper for a city."""
    try:
        from scrapers.combined_scraper import run_combined_scraper as run_scraper
        return run_scraper(city)
    except ImportError:
        print("Combined scraper not found, falling back to Zomato scraper")
        return run_scraper_if_needed(f"https://www.zomato.com/{city}")