import time
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains


CACHE_FILE = "swiggy_cache.json"
MAX_THREADS = 4
RETRY_COUNT = 3
TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"


class SwiggyScraper:
    def __init__(self, headless: bool = False, cache: bool = True):
        """
        Initialize the Swiggy scraper.

        Args:
            headless: Run browser in headless mode (no GUI)
            cache: Use caching to store and retrieve results
        """
        self.headless = headless
        self.use_cache = cache
        self.cache = self._load_cache() if cache else {}
        self.driver = None

    def _load_cache(self) -> Dict:
        """Load cached restaurant data if available."""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Cache loading error: {e}")
        return {}

    def _save_cache(self):
        """Save scraped data to cache file."""
        if not self.use_cache:
            return
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Cache saving error: {e}")

    def _initialize_driver(self):
        """Set up and return a Chrome WebDriver instance."""
        chrome_opts = Options()

        if self.headless:
            chrome_opts.add_argument("--headless=new")

        chrome_opts.add_argument("--window-size=1920,1080")
        chrome_opts.add_argument("--disable-gpu")
        chrome_opts.add_argument("--disable-extensions")
        chrome_opts.add_argument(f"user-agent={USER_AGENT}")
        chrome_opts.add_experimental_option(
            "excludeSwitches", ["enable-automation"])
        chrome_opts.add_experimental_option("useAutomationExtension", False)
        chrome_opts.add_argument("--no-sandbox")
        chrome_opts.add_argument("--disable-dev-shm-usage")

        service = Service()
        return webdriver.Chrome(service=service, options=chrome_opts)

    def get_restaurant_urls(self, city_url: str, scroll_pause: float = 1.5, max_show_more_clicks: int = 10) -> List[str]:
        """
        Extract all restaurant URLs from a Swiggy city page.

        Args:
            city_url: URL of the city page on Swiggy
            scroll_pause: Time to wait between actions in seconds
            max_show_more_clicks: Maximum number of "Show more" clicks to perform

        Returns:
            List of restaurant URLs
        """
        print(f"Collecting restaurant URLs from {city_url}...")

        cache_key = f"urls_{city_url.split('/')[-1]}"
        if self.use_cache and cache_key in self.cache:
            print(f"Using {len(self.cache[cache_key])} cached URLs")
            return self.cache[cache_key]

        self.driver = self._initialize_driver()
        restaurant_urls = []

        try:
            self.driver.get(city_url)
            time.sleep(scroll_pause * 2)  # Initial load

            try:
                location_popup = self.driver.find_element(
                    By.XPATH, "//button[contains(text(), 'DETECT') or contains(text(), 'detect')]")
                location_popup.click()
                time.sleep(2)
            except:
                pass

            self.driver.execute_script("window.scrollTo(0, 500);")
            time.sleep(scroll_pause)

            show_more_clicks = 0
            while show_more_clicks < max_show_more_clicks:

                self._extract_restaurant_urls_from_page(restaurant_urls)

                try:
                    show_more_button = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable(
                            (By.CSS_SELECTOR, "div[data-testid='restaurant_list_show_more']"))
                    )
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center'});", show_more_button)
                    time.sleep(1)
                    show_more_button.click()
                    print(
                        f"Clicked 'Show more' button ({show_more_clicks + 1}/{max_show_more_clicks})")
                    # Wait for new content to load
                    time.sleep(scroll_pause * 2)
                    show_more_clicks += 1
                except Exception as e:
                    print(
                        f"No more 'Show more' button found or error: {str(e)}")
                    break

            self._extract_restaurant_urls_from_page(restaurant_urls)

            # Remove duplicates
            restaurant_urls = list(set(restaurant_urls))
            print(f"\nFound {len(restaurant_urls)} restaurant URLs")

            # Cache URLs
            if self.use_cache:
                self.cache[cache_key] = restaurant_urls
                self._save_cache()

            return restaurant_urls

        finally:
            if self.driver:
                self.driver.quit()
                self.driver = None

    def _extract_restaurant_urls_from_page(self, restaurant_urls: List[str]):
        """Helper method to extract restaurant URLs from the current page."""
        try:

            restaurant_cards = self.driver.find_elements(
                By.CSS_SELECTOR, "div[data-testid='restaurant_list_card']")

            for card in restaurant_cards:
                try:

                    link_element = card.find_element(By.TAG_NAME, "a")
                    href = link_element.get_attribute("href")

                    if href and "swiggy.com/city" in href:
                        restaurant_urls.append(href)
                except Exception:
                    continue

            print(f"Found {len(restaurant_urls)} restaurant URLs so far")
        except Exception as e:
            print(f"Error extracting restaurant URLs: {str(e)}")

    def scrape_restaurant(self, url: str) -> Dict[str, Any]:
        """
        Scrape details from a single Swiggy restaurant page.

        Args:
            url: URL of the restaurant page on Swiggy

        Returns:
            Dictionary containing restaurant details
        """

        if self.use_cache and url in self.cache:
            return self.cache[url]

        restaurant_data = {"url": url,
                           "scraped_at": datetime.now().isoformat(),
                           "source": "swiggy",
                        }
        retry_count = 0

        while retry_count < RETRY_COUNT:
            driver = None
            try:

                driver = self._initialize_driver()
                driver.get(url)

                WebDriverWait(driver, TIMEOUT).until(
                    EC.presence_of_element_located((By.TAG_NAME, "h1"))
                )

                name = self._safe_extract(driver, [
                    (By.CSS_SELECTOR, "h1.sc-aXZVg.gONLwH"),
                ])
                restaurant_data["name"] = name

                try:
                    rating_text = self._safe_extract(driver, [
                        (By.CSS_SELECTOR, "div.sc-aXZVg.bTHhpu"),
                    ])
                    if rating_text and rating_text != "Not found":

                        rating_match = re.search(
                            r"(\d+\.\d+|\d+)", rating_text)
                        if rating_match:
                            restaurant_data["rating"] = rating_match.group(1)

                            restaurant_data["rating_details"] = rating_text
                        else:
                            restaurant_data["rating"] = rating_text
                    else:
                        restaurant_data["rating"] = "Not available"
                except Exception as e:
                    restaurant_data["rating"] = "Not available"

                try:
                    cuisine_elements = driver.find_elements(
                        By.CSS_SELECTOR, "div.sc-aXZVg.bPYyBR.sc-iLsKjm.bYSiwj")
                    if cuisine_elements:
                        cuisine_list = [elem.text.strip()
                                        for elem in cuisine_elements]

                        cuisine = ", ".join([c for c in cuisine_list if c])
                        restaurant_data["cuisine"] = cuisine

                    else:
                        restaurant_data["cuisine"] = "Not found"
                except Exception as e:
                    restaurant_data["cuisine"] = "Not found"
                    print(f"Error extracting cuisine: {e}")

                # Extract address
                address = self._safe_extract(driver, [
                    # Target the specific structure you shared
                    (By.XPATH,
                     "//div[contains(text(), 'Outlet')]/following-sibling::div[contains(@class, 'sc-aXZVg')]"),
                    (By.CSS_SELECTOR, "div.sc-aXZVg.sc-llILlE"),
                    (By.CSS_SELECTOR, "div.sc-aXZVg[class*='kYaB']"),
                    (By.XPATH, "//div[contains(@class, 'address')]"),
                    (By.XPATH, "//div[contains(@class, 'location')]")
                ])
                restaurant_data["address"] = address

                restaurant_data["menu_items"] = self._extract_menu_items(
                    driver)

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

                if retry_count < RETRY_COUNT:
                    time.sleep(2 ** retry_count)

        if self.use_cache:
            self.cache[url] = restaurant_data

            self._save_cache()

        return restaurant_data

    def _safe_extract(self, driver, selectors) -> str:
        """Try multiple selectors and return the first successful result."""
        for selector_type, selector in selectors:
            try:
                element = driver.find_element(selector_type, selector)
                return element.text.strip()
            except NoSuchElementException:
                continue
        return "Not found"

    def _extract_menu_items(self, driver) -> Dict[str, List[Dict]]:
        """Extract menu items from the restaurant page with categories and details."""
        menu_data = {}

        time.sleep(5)

        try:
            menu_tabs = driver.find_elements(
                By.XPATH, "//a[contains(text(), 'Menu') or contains(text(), 'Order')]")
            if menu_tabs:
                print("Found menu tab, clicking it...")
                menu_tabs[0].click()
                time.sleep(3)
        except Exception:
            pass

        for scroll_pos in [300, 800, 1500, 2200]:
            try:
                driver.execute_script(f"window.scrollTo(0, {scroll_pos});")
                time.sleep(1)
            except Exception:
                pass

        try:
            with open("swiggy_debug.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print("Saved HTML to swiggy_debug.html for debugging")
        except Exception as e:
            print(f"Could not save debug HTML: {e}")

        try:

            category_headers = driver.find_elements(
                By.CSS_SELECTOR, "h3.sc-aXZVg.kSWsUU")

            # Process categories if found
            if category_headers and len(category_headers) > 0:
                for header in category_headers:
                    try:
                        category_name = header.text.strip()
                        if not category_name:
                            continue

                        parent_container = driver.execute_script(
                            "return arguments[0].nextElementSibling;", header)

                        if parent_container:

                            items = []

                            name_elements = parent_container.find_elements(By.CSS_SELECTOR,
                                                                           "div.sc-aXZVg.eqSzsP.sc-eeDRCY.dwSeRx")

                            for name_elem in name_elements:
                                try:
                                    item = {}

                                    item["name"] = name_elem.text.strip()

                                    item_container = name_elem.find_element(
                                        By.XPATH, "./ancestor::div[position()=3]")

                                    try:
                                        price_elem = item_container.find_element(
                                            By.CSS_SELECTOR, "div.sc-iHGNWf.eRzEaY")
                                        item["price"] = "₹" + \
                                            price_elem.text.strip()
                                    except NoSuchElementException:
                                        item["price"] = "Price not available"

                                    if item.get("name"):
                                        items.append(item)

                                except Exception as e:
                                    print(f"Error processing item: {e}")
                                    continue

                            if items:
                                menu_data[category_name] = items
                                print(
                                    f"Added {len(items)} items to category '{category_name}'")

                    except Exception as e:
                        print(f"Error processing category: {e}")
                        continue

            if not menu_data:
                print("No categories found, trying direct item extraction...")
                items = []

                name_elements = driver.find_elements(
                    By.CSS_SELECTOR, "div.sc-aXZVg.eqSzsP.sc-eeDRCY.dwSeRx")

                for name_elem in name_elements:
                    try:
                        item = {}

                        item["name"] = name_elem.text.strip()

                        item_container = name_elem.find_element(
                            By.XPATH, "./ancestor::div[position()=3]")

                        try:
                            price_elem = item_container.find_element(
                                By.CSS_SELECTOR, "div.sc-iHGNWf.eRzEaY")
                            item["price"] = "₹" + price_elem.text.strip()
                        except NoSuchElementException:
                            item["price"] = "Price not available"

                        if item.get("name"):
                            items.append(item)

                    except Exception as e:
                        continue

                if items:
                    menu_data["Menu Items"] = items

        except Exception as e:
            print(f"Menu extraction error: {str(e)}")

        total_items = sum(len(items) for items in menu_data.values())

        return menu_data

    def scrape_all_restaurants(self, city_url: str) -> List[Dict]:
        """
        Get and scrape all restaurant data from a city page.

        Args:
            city_url: URL of the city page on Swiggy

        Returns:
            List of dictionaries containing restaurant data
        """
        restaurant_urls = self.get_restaurant_urls(city_url)

        if not restaurant_urls:
            print("No restaurant URLs found.")
            return []

        print(f"Scraping {len(restaurant_urls)} restaurants...")
        results = []

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            future_to_url = {executor.submit(
                self.scrape_restaurant, url): url for url in restaurant_urls}

            completed = 0
            for future in future_to_url:
                url = future_to_url[future]
                try:
                    data = future.result()
                    results.append(data)
                except Exception as exc:
                    print(f"{url} generated an exception: {exc}")

                completed += 1
                print(
                    f"Progress: {completed}/{len(restaurant_urls)} restaurants scraped", end='\r')

        print("\nScraping completed!")
        return results


if __name__ == "__main__":
    # Example usage
    scraper = SwiggyScraper(headless=True, cache=True)

    # Get restaurant data from Roorkee or any other city
    city_url = "https://www.swiggy.com/city/roorkee"
    restaurants = scraper.scrape_all_restaurants(city_url)

    # Save to JSON
    output_file = f"swiggy_{city_url.split('/')[-1]}_{datetime.now().strftime('%Y%m%d')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(restaurants, f, indent=2, ensure_ascii=False)

    print(f"Saved data for {len(restaurants)} restaurants to {output_file}")
