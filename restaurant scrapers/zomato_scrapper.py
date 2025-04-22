import time
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

# Configuration
CACHE_FILE = "zomato_cache.json"
MAX_THREADS = 4  # Adjust based on your system capability
RETRY_COUNT = 3
TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

class ZomatoScraper:
    def __init__(self, headless: bool = False, cache: bool = True):
        """
        Initialize the Zomato scraper.
        
        Args:
            headless: Run browser in headless mode (no GUI)
            cache: Use caching to store and retrieve results
        """
        self.headless = headless
        self.use_cache = cache
        self.cache = self._load_cache() if cache else {"cities": {}, "restaurants": {}}
        self.driver = None
        
        # Initialize cache structure if needed
        if "cities" not in self.cache:
            self.cache["cities"] = {}
        if "restaurants" not in self.cache:
            self.cache["restaurants"] = {}
        
    def _load_cache(self) -> Dict:
        """Load cached restaurant data if available."""
        try:
            if os.path.exists(CACHE_FILE):
                print(f"Loading cache from {CACHE_FILE}")
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Initialize sections if they don't exist
                    if "cities" not in cache_data:
                        cache_data["cities"] = {}
                    if "restaurants" not in cache_data:
                        cache_data["restaurants"] = {}
                    return cache_data
        except Exception as e:
            print(f"Cache loading error: {e}")
        return {"cities": {}, "restaurants": {}}
    
    def _save_cache(self):
        """Save scraped data to cache file."""
        if not self.use_cache:
            return
        try:
            print(f"Saving cache to {CACHE_FILE}")
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
            print("Cache saved successfully")
        except Exception as e:
            print(f"Cache saving error: {e}")
    
    def _extract_city_name(self, city_url: str) -> str:
        """Extract city name from URL for better caching."""
        # Extract city name from url like "https://www.zomato.com/roorkee"
        city_name = city_url.rstrip('/').split('/')[-1]
        if not city_name:
            city_name = "unknown"
        return city_name
    
    def _initialize_driver(self):
        """Set up and return a Chrome WebDriver instance."""
        chrome_opts = Options()
        # Always run in headless mode
        chrome_opts.add_argument("--headless=new")
        
        chrome_opts.add_argument("--window-size=1920,1080")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--disable-extensions")
        chrome_opts.add_argument(f"user-agent={USER_AGENT}")
        chrome_opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_opts.add_experimental_option("useAutomationExtension", False)
        # Additional options to ensure headless works properly
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")
        
        service = Service()
        return webdriver.Chrome(service=service, options=chrome_opts)
    
    def get_restaurant_urls(self, city_url: str, scroll_pause: float = 1.5, max_scrolls: int = 30) -> List[str]:
        """
        Extract all restaurant URLs from a Zomato city page.
        
        Args:
            city_url: URL of the city page on Zomato (e.g., "https://www.zomato.com/roorkee")
            scroll_pause: Time to wait between scrolls in seconds
            max_scrolls: Maximum number of scrolls to perform
            
        Returns:
            List of restaurant URLs ending with "/order"
        """
        print(f"Collecting restaurant URLs from {city_url}...")
        
        # Extract city name for more robust caching
        city_name = self._extract_city_name(city_url)
        print(f"Determined city name: {city_name}")
        
        # Check if URLs are cached
        if self.use_cache and city_name in self.cache["cities"]:
            cached_urls = self.cache["cities"][city_name]
            print(f"Using {len(cached_urls)} cached URLs for {city_name}")
            return cached_urls
        
        self.driver = self._initialize_driver()
        restaurant_urls = []
        
        try:
            self.driver.get(city_url)
            time.sleep(scroll_pause)  # Initial load
            
            # Create progress indicator
            for scroll in range(max_scrolls):
                print(f"Scrolling page: {scroll+1}/{max_scrolls}", end='\r')
                
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(scroll_pause)
                
                # Check if we've reached the bottom
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                last_height = self.driver.execute_script("return window.lastScrollHeight || 0")
                self.driver.execute_script("window.lastScrollHeight = arguments[0]", new_height)
                
                if new_height == last_height:
                    print(f"\nReached bottom of page after {scroll+1} scrolls")
                    break
            
            # Extract restaurant URLs
            anchors = self.driver.find_elements(By.TAG_NAME, "a")
            for a in anchors:
                try:
                    href = a.get_attribute("href")
                    if href and href.endswith("/order"):
                        restaurant_urls.append(href)
                    elif href and href.endswith("/info"):
                        restaurant_urls.append(href.replace("/info", "/order"))
                except Exception:
                    continue
            
            restaurant_urls = list(set(restaurant_urls))  # Remove duplicates
            print(f"\nFound {len(restaurant_urls)} restaurant URLs")
            
            # Cache URLs by city name
            if self.use_cache:
                self.cache["cities"][city_name] = restaurant_urls
                self._save_cache()
                print(f"Cached {len(restaurant_urls)} URLs for {city_name}")
            
            return restaurant_urls
            
        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None
    
    def scrape_restaurant(self, url: str) -> Dict[str, Any]:
        """
        Scrape details from a single Zomato restaurant page.
        
        Args:
            url: URL of the restaurant page on Zomato
            
        Returns:
            Dictionary containing restaurant details
        """
        # Generate a cache key that includes the URL
        cache_key = url
        
        # Check cache first
        if self.use_cache and cache_key in self.cache["restaurants"]:
            print(f"Using cached data for {url}")
            return self.cache["restaurants"][cache_key]
        
        # Extract city name from URL for data organization
        city_name = self._extract_city_from_url(url)
        
        restaurant_data = {
            "url": url, 
            "scraped_at": datetime.now().isoformat(),
            "city": city_name
        }
        retry_count = 0
        
        while retry_count < RETRY_COUNT:
            driver = None
            try:
                # Use a new driver for each restaurant
                driver = self._initialize_driver()
                driver.get(url)
                
                # Wait for page to load
                WebDriverWait(driver, TIMEOUT).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "h1"))
                )
                
                # Extract restaurant name (try multiple selectors)
                name = self._safe_extract(driver, [
                    (By.CSS_SELECTOR, "h1.sc-7kepeu-0"),
                    (By.CSS_SELECTOR, "h1[data-testid='header-title']"),
                    (By.TAG_NAME, "h1")
                ])
                restaurant_data["name"] = name
                
                # Extract ratings
                try:
                    rating_blocks = driver.find_elements(By.CSS_SELECTOR, ".sc-1q7bklc-5, [data-testid='rating-widget']")
                    ratings = {}
                    for block in rating_blocks:
                        try:
                            score = block.find_element(By.CSS_SELECTOR, ".sc-1q7bklc-1, [data-testid='rating-score']").text
                            label = block.find_element(By.CSS_SELECTOR, ".sc-1q7bklc-9, [data-testid='rating-type']").text
                            ratings[label] = score
                        except NoSuchElementException:
                            continue
                    restaurant_data["ratings"] = ratings
                except Exception as e:
                    restaurant_data["ratings"] = {"error": str(e)}
                
                # Extract cuisine type
                cuisine = self._safe_extract(driver, [
                    (By.CSS_SELECTOR, "a.sc-eXNvrr"),
                    (By.CSS_SELECTOR, "[data-testid='cuisine-type']"),
                    (By.XPATH, "//span[contains(text(), 'Cuisine')]/following-sibling::*")
                ])
                restaurant_data["cuisine"] = cuisine
                
                # Extract address
                address = self._safe_extract(driver, [
                    (By.CSS_SELECTOR, "div.sc-clNaTc"),
                    (By.CSS_SELECTOR, "[data-testid='restaurant-address']"),
                    (By.XPATH, "//span[contains(text(), 'Address')]/following-sibling::*")
                ])
                restaurant_data["address"] = address
                
                # Extract phone number
                try:
                    phone_elem = driver.find_element(
                        By.XPATH,
                        "//i[contains(@class,'sc-rbbb40-1') and @color='#ef4f5f']/ancestor::a"
                    )
                    phone = phone_elem.get_attribute("href").replace("tel:", "")
                    restaurant_data["phone"] = phone
                except Exception:
                    restaurant_data["phone"] = "Not available"
                
                # Try to get order online URL
                try:
                    order_link = driver.find_element(
                        By.XPATH,
                        "//span[@title='Order Online']/ancestor::a"
                    ).get_attribute("href")
                    restaurant_data["order_link"] = order_link
                except Exception:
                    restaurant_data["order_link"] = url
                
                # Check restaurant ordering status - MOVED FROM MENU METHOD
                try:
                    closed_indicators = [
                        "//div[contains(text(), 'Currently closed for online ordering')]",
                        "//div[contains(@class, 'subtitle') and contains(text(), 'closed')]",
                        "//div[contains(text(), 'Online ordering is only supported')]",
                        "//div[contains(text(), 'temporarily closed')]"
                    ]
                    
                    for indicator in closed_indicators:
                        if driver.find_elements(By.XPATH, indicator):
                            restaurant_data["status"] = "closed"
                            print(f"Restaurant is closed for online ordering: {url}")
                            break
                    else:
                        restaurant_data["status"] = "open"
                except Exception:
                    restaurant_data["status"] = "unknown"
                
                # Always try to extract menu items regardless of status
                restaurant_data["menu_items"] = self._extract_menu_items(driver, url)
                
                # If we got this far, we've succeeded
                break
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Error scraping {url}: {str(e)}"
                print(error_msg)
                if retry_count >= RETRY_COUNT:
                    restaurant_data["error"] = error_msg
            
            finally:
                if driver:
                    driver.quit()
                    
                # Exponential backoff
                if retry_count < RETRY_COUNT:
                    time.sleep(2 ** retry_count)
        
        # Cache the result with structured key
        if self.use_cache:
            self.cache["restaurants"][cache_key] = restaurant_data
            # Save after each restaurant to prevent data loss
            self._save_cache()
            
        return restaurant_data
    
    def _extract_city_from_url(self, url: str) -> str:
        """Extract city name from restaurant URL."""
        # Extract city from URLs like https://www.zomato.com/roorkee/restaurant-name/order
        match = re.search(r'zomato\.com/([^/]+)', url)
        if match:
            return match.group(1)
        return "unknown"
    
    def _safe_extract(self, driver, selectors) -> str:
        """Try multiple selectors and return the first successful result."""
        for selector_type, selector in selectors:
            try:
                element = driver.find_element(selector_type, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        return "Not found"
    
    def _extract_menu_items(self, driver, restaurant_url: str) -> Dict[str, List[Dict]]:
        """Extract menu items from the restaurant page with categories and details."""
        menu_data = {}
        
        # Navigate to menu tab if needed and trigger content loading
        if "/order" not in driver.current_url:
            try:
                # Find and click menu tab
                menu_tabs = driver.find_elements(By.XPATH, 
                    "//a[contains(@href, '/order') or contains(text(), 'Order Online')]")
                if menu_tabs:
                    for tab in menu_tabs:
                        tab.click()
                        time.sleep(3)
                        break
            except Exception:
                pass
        
        # Multiple scrolling to trigger lazy loading
        for scroll_pos in [300, 800, 1500]:
            try:
                driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                time.sleep(1)
            except Exception:
                pass
        
        # Extract menu categories - using more robust XPath patterns and attribute selectors
        try:
            # First approach: Find categories by their structure
            category_headers = driver.find_elements(By.XPATH, 
                "//h4[contains(@class, 'sc-') or contains(@class, 'category') or contains(@class, 'head')]")
                
            # If that fails, try an alternative approach
            if not category_headers:
                category_headers = driver.find_elements(By.XPATH, 
                    "//div[contains(@class, 'menu') or contains(@class, 'category')]//h4")
                
            # If that still fails, try finding them by position/structure
            if not category_headers:
                category_headers = driver.find_elements(By.XPATH,
                    "//div[.//div[contains(@class, 'dish') or contains(@class, 'menu-item')]]/preceding-sibling::h4")
            
            # Process categories if found
            if category_headers and len(category_headers) > 0:
                for header in category_headers:
                    try:
                        category_name = header.text.strip()
                        if not category_name:
                            continue
                        
                        # Find item container - use a structure-based approach
                        item_container = None
                        
                        # Approach 1: Look for following sibling that contains menu items
                        try:
                            # First try direct sibling
                            item_container = driver.execute_script(
                                "return arguments[0].nextElementSibling;", header)
                        except Exception:
                            pass
                        
                        # Approach 2: Look for parent then sibling
                        if not item_container:
                            try:
                                parent = header.find_element(By.XPATH, "./..")
                                # Look for any div that follows the header within this parent
                                item_container = parent.find_element(By.XPATH, 
                                    "./following-sibling::div[1]")
                            except Exception:
                                pass
                        
                        # Approach 3: Look based on content
                        if not item_container:
                            try:
                                # Find the first container after this header that contains items
                                item_container = driver.find_element(By.XPATH,
                                    f"//h4[contains(text(), '{category_name}')]/following::div[contains(., '₹') or .//div[@type='veg']][1]")
                            except Exception:
                                pass
                        
                        # Extract items from container
                        if item_container:
                            items = self._extract_items_from_container(item_container)
                            if items:
                                menu_data[category_name] = items
                    
                    except Exception as e:
                        print(f"Error processing category: {str(e)}")
                        continue
            
            # Fallback: direct extraction without categories
            if not menu_data:
                # Try to find any menu items on the page
                items = []
                
                # Find items by their structure rather than specific classes
                item_elements = driver.find_elements(By.XPATH, 
                    "//div[.//h4 and .//span[contains(text(), '₹')]]")
                
                if not item_elements:
                    item_elements = driver.find_elements(By.XPATH,
                        "//div[.//h4][.//div[@type='veg' or @type='non-veg']]")
                    
                if not item_elements:
                    # Last resort: find anything that looks like a menu item
                    item_elements = driver.find_elements(By.XPATH,
                        "//div[contains(., '₹') and .//h4]")
                
                for item_elem in item_elements: 
                    item = self._extract_item_details(item_elem)
                    if item and item.get("name"):
                        items.append(item)
                
                if items:
                    menu_data["Menu Items"] = items
                    
        except Exception as e:
            print(f"Menu extraction error: {str(e)}")
        
        total_items = sum(len(items) for items in menu_data.values())
        print(f"Extracted {total_items} menu items from {len(menu_data)} categories")
        
        return menu_data
    
    def _extract_items_from_container(self, container) -> List[Dict]:
        """Extract menu items from a container using structure-based selectors."""
        items = []
        
        # First try to find items by their structure
        item_elements = container.find_elements(By.XPATH, 
            ".//div[.//h4 and (.//span[contains(text(), '₹')] or .//div[@type])]")
        
        # If that fails, try a more general approach
        if not item_elements:
            item_elements = container.find_elements(By.XPATH, 
                ".//div[.//h4]")  # Limit to avoid false positives
        
        for item_elem in item_elements:
            item = self._extract_item_details(item_elem)
            if item and item.get("name"):
                items.append(item)
        
        return items
    
    def _extract_item_details(self, item_elem) -> Dict:
        """Extract details from a menu item element using structure not classes."""
        item = {}
        
        try:
            # Name - find any h4 heading inside this item
            try:
                name_elem = item_elem.find_element(By.XPATH, ".//h4")
                item["name"] = name_elem.text.strip()
            except NoSuchElementException:
                return None  # No name, no item
            
            # Price - find anything with rupee symbol
            try:
                price_elem = item_elem.find_element(By.XPATH, 
                    ".//*[contains(text(), '₹') or contains(text(), 'Rs')]")
                item["price"] = price_elem.text.strip()
            except NoSuchElementException:
                # Try another approach for price
                try:
                    price_elem = item_elem.find_element(By.XPATH, ".//span[number(text()) > 0]")
                    price = price_elem.text.strip()
                    if price.isdigit() or price.replace('.', '', 1).isdigit():
                        item["price"] = f"₹{price}"
                    else:
                        item["price"] = "Price not available"
                except NoSuchElementException:
                    item["price"] = "Price not available"
            
            # Description - any paragraph tag
            try:
                desc_elem = item_elem.find_element(By.XPATH, ".//p")
                desc = desc_elem.text.strip()
                if desc:
                    item["description"] = desc
            except NoSuchElementException:
                item["description"] = ""
            
            # Veg/Non-veg indicator - using type attribute
            try:
                veg_indicator = item_elem.find_element(By.XPATH, ".//div[@type]")
                veg_type = veg_indicator.get_attribute("type")
                item["type"] = "veg" if "veg" in veg_type.lower() else "non-veg"
            except NoSuchElementException:
                # Try another approach - look for green/red icons
                try:
                    veg_icon = item_elem.find_element(By.XPATH, ".//*[contains(@fill, '#3AB757')]")
                    item["type"] = "veg"
                except NoSuchElementException:
                    try:
                        non_veg_icon = item_elem.find_element(By.XPATH, ".//*[contains(@fill, '#F63440')]")
                        item["type"] = "non-veg"
                    except NoSuchElementException:
                        item["type"] = "unknown"
            
            return item
            
        except Exception:
            return None
    
    def _extract_items_from_container(self, container) -> List[Dict]:
        """Helper method to extract menu items from a container."""
        items = []
        # Target specific dish containers
        item_elements = container.find_elements(By.CSS_SELECTOR, 
            ".sc-MJoYu, .bEimze, [class*='dishContainer'], [class*='menuItem']")
        
        # If no items found, try broader selectors
        if not item_elements:
            item_elements = container.find_elements(By.CSS_SELECTOR, 
                "[class*='dish'], [class*='menu-item'], [class*='food-item']")
        
        for item_elem in item_elements: 
            try:
                item = {}
                
                # Name - using exact classes from example
                name_found = False
                for selector in ["h4.sc-eNaZKA", ".cyhCrC", "[class*='itemName']", "h4", "h3"]:
                    try:
                        name_elem = item_elem.find_element(By.CSS_SELECTOR, selector)
                        name = name_elem.text.strip()
                        if name:
                            item["name"] = name
                            name_found = True
                            break
                    except:
                        continue
                
                if not name_found:
                    continue
                
                # Price - using exact classes from example
                try:
                    for selector in ["span.sc-17hyc2s-1", ".cCiQWA", "[class*='price']"]:
                        price_elem = item_elem.find_element(By.CSS_SELECTOR, selector)
                        price = price_elem.text.strip()
                        if price and ('₹' in price or 'Rs' in price):
                            item["price"] = price
                            break
                except:
                    item["price"] = "Price not available"
                
                # Description - using exact classes from example
                try:
                    for selector in ["p.sc-cmUVTD", ".MxLPh", "[class*='description']", "p"]:
                        desc_elem = item_elem.find_element(By.CSS_SELECTOR, selector)
                        desc = desc_elem.text.strip()
                        if desc:
                            item["description"] = desc
                            break
                except:
                    item["description"] = ""
                
                # Veg/Non-veg indicator
                try:
                    veg_indicator = item_elem.find_element(By.CSS_SELECTOR, 
                        "div.sc-dPgCzP, .hFYiSp, [class*='itemType']")
                    item["type"] = "veg" if "veg" in veg_indicator.get_attribute("type") else "non-veg"
                except:
                    item["type"] = "unknown"
                
                # Add item if we have a name
                if "name" in item:
                    items.append(item)
                    
            except:
                continue
        
        return items
        
    def scrape_all_restaurants(self, city_url: str) -> List[Dict]:
        """
        Get and scrape all restaurant data from a city page.
        
        Args:
            city_url: URL of the city page on Zomato
            
        Returns:
            List of dictionaries containing restaurant data
        """
        city_name = self._extract_city_name(city_url)
        restaurant_urls = self.get_restaurant_urls(city_url)
        
        if not restaurant_urls:
            print("No restaurant URLs found.")
            return []
        
        print(f"Scraping {len(restaurant_urls)} restaurants in {city_name}...")
        start_time = datetime.now()
        results = []
        
        # Use ThreadPoolExecutor for parallel scraping
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_url = {executor.submit(self.scrape_restaurant, url): url for url in restaurant_urls}
            
            completed = 0
            successful = 0
            failed = 0
            
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    data = future.result()
                    results.append(data)
                    successful += 1
                except Exception as exc:
                    print(f"\n{url} generated an exception: {exc}")
                    failed += 1
                
                completed += 1
                
                # More informative progress report
                percent = (completed / len(restaurant_urls)) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Print progress every time or at certain intervals (e.g., every 5 restaurants)
                if completed == 1 or completed % 5 == 0 or completed == len(restaurant_urls):
                    print(f"Progress: {completed}/{len(restaurant_urls)} restaurants scraped ({percent:.1f}%) - Success: {successful}, Failed: {failed} - Elapsed: {elapsed:.1f}s", end='\r')
        
        # Print a newline after progress reporting to prevent overwriting
        print("\nScraping completed!")
        
        # Calculate and display statistics
        total_time = (datetime.now() - start_time).total_seconds()
        print(f"Summary: {successful} successful, {failed} failed in {total_time:.1f} seconds")
        
        # Save city-specific results to a separate file
        output_file = f"zomato_{city_name}_{datetime.now().strftime('%Y%m%d')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Saved data for {len(results)} restaurants to {output_file}")
        return results

if __name__ == "__main__":
    # Example usage
    scraper = ZomatoScraper(headless=True, cache=True)
    
    # Specify city URL
    city_url = "https://www.zomato.com/ncr"  # Change to your desired city
    restaurants = scraper.scrape_all_restaurants(city_url)
    
    print(f"Completed scraping {len(restaurants)} restaurants from {city_url}")