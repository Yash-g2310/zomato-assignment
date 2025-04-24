import argparse
import logging
import os
import sys
import time
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config
from config import get_latest_data_file

def run_scraper(city=None, force=False):
    """Run the scraper for a given city or all default cities."""
    from scrapers.combined_scraper import run_combined_scraper, scrape_multiple_cities, DEFAULT_CITIES
    
    if city:
        # Run for a specific city
        return run_combined_scraper(city, force)
    else:
        # Run for all default cities
        logger.info(f"Running scraper for all default cities: {', '.join(DEFAULT_CITIES)}")
        return scrape_multiple_cities(force=force)

def run_knowledge_base(data_file=None):
    """Run knowledge base with specified data file or latest available."""
    from knowledge_base import RestaurantKnowledgeBase, main as kb_main
    
    if data_file is None:
        data_file = get_latest_data_file()
        if not data_file:
            logger.error("No data file found. Please run the scraper first.")
            return None
    
    kb = RestaurantKnowledgeBase()
    kb.load_from_json(data_file)
    return kb

def run_rag_demo(data_file=None, lightweight=True):
    """Run the RAG demo with the specified data file."""
    from restaurant_rag import demo_rag
    
    if data_file is None:
        data_file = get_latest_data_file()
        if not data_file:
            logger.error("No data file found. Please run the scraper first.")
            return
    
    demo_rag()

def test_rag_system(data_file=None, lightweight=True):
    """Test the RAG system with pre-defined queries to validate functionality."""
    from restaurant_rag import RestaurantRAG, find_latest_json_file
    
    if data_file is None:
        try:
            data_file = find_latest_json_file()
            logger.info(f"Using data file: {data_file}")
        except FileNotFoundError:
            logger.error("No data file found. Please run the scraper first.")
            return False
    
    logger.info("Initializing RAG system for testing...")
    try:
        rag = RestaurantRAG(data_file, use_lightweight=lightweight)
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False
    
    # Define test queries covering different aspects
    test_queries = [
        "What restaurants are in Roorkee?",
        "Where can I find vegetarian food?",
        "Which restaurant serves butter chicken?",
        "What is the price range for pizzas?",
        "Is there a bakery open now?"
    ]
    
    logger.info("Running test queries...")
    success_count = 0
    
    for i, query in enumerate(test_queries, 1):
        try:
            logger.info(f"Test {i}: '{query}'")
            start_time = time.time()
            result = rag.query(query)
            elapsed = time.time() - start_time
            
            if result and "answer" in result and len(result["answer"]) > 20:
                logger.info(f"✓ Query {i} successful ({elapsed:.2f}s)")
                logger.info(f"  Type: {result.get('query_type', 'unknown')}")
                logger.info(f"  Answer: {result['answer'][:100]}...")
                success_count += 1
            else:
                logger.warning(f"✗ Query {i} returned insufficient data")
                logger.warning(f"  Result: {result}")
        except Exception as e:
            logger.error(f"✗ Query {i} failed with error: {e}")
    
    success_rate = (success_count / len(test_queries)) * 100
    logger.info(f"Test results: {success_count}/{len(test_queries)} queries successful ({success_rate:.1f}%)")
    
    if success_rate > 80:
        logger.info("RAG system is working well!")
        return True
    elif success_rate > 50:
        logger.warning("RAG system is working but may have some issues.")
        return True
    else:
        logger.error("RAG system is not functioning correctly. Please check the implementation.")
        return False

def run_streamlit():
    """Launch the Streamlit app."""
    import subprocess
    import os
    streamlit_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    
    if not os.path.exists(streamlit_path):
        logger.error(f"Streamlit app not found at {streamlit_path}")
        fallback_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
        if os.path.exists(fallback_path):
            logger.info(f"Found streamlit app at original location: {fallback_path}")
            streamlit_path = fallback_path
        else:
            logger.error("No streamlit app found. Please check the installation.")
            return
    
    logger.info(f"Launching Streamlit app from: {streamlit_path}")
    
    # Set environment variable to disable file watching - this prevents the PyTorch error
    env = os.environ.copy()
    env["STREAMLIT_WATCH_FOR_CHANGES"] = "false"
    
    # Use the modified environment when launching Streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path], env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant Knowledge System")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Run the scraper")
    scraper_parser.add_argument("city", help="City to scrape (e.g., roorkee, ncr)")
    scraper_parser.add_argument("--city", help="City to scrape (e.g., roorkee, ncr). If not provided, all default cities will be scraped.")
    
    # Knowledge base command
    kb_parser = subparsers.add_parser("kb", help="Run knowledge base queries")
    kb_parser.add_argument("--data", help="Path to data file (will use latest if not specified)")
    
    # RAG demo command
    rag_parser = subparsers.add_parser("rag", help="Run RAG demo")
    rag_parser.add_argument("--data", help="Path to data file (will use latest if not specified)")
    rag_parser.add_argument("--full", action="store_true", help="Use full-size model instead of lightweight")
    
    # RAG test command
    rag_test_parser = subparsers.add_parser("test-rag", help="Test RAG system functionality")
    rag_test_parser.add_argument("--data", help="Path to data file (will use latest if not specified)")
    rag_test_parser.add_argument("--full", action="store_true", help="Use full-size model instead of lightweight")
    
    # Streamlit command
    streamlit_parser = subparsers.add_parser("ui", help="Launch Streamlit UI")
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        run_scraper(args.city, args.force)
    elif args.command == "kb":
        run_knowledge_base(args.data)
    elif args.command == "rag":
        run_rag_demo(args.data, not args.full)
    elif args.command == "test-rag":
        test_rag_system(args.data, not args.full)
    elif args.command == "ui":
        run_streamlit()
    else:
        parser.print_help()