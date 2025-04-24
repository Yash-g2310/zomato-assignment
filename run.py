import argparse
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import config
from config import get_latest_data_file

def run_scraper(city, force=False):
    """Run the scraper for a given city."""
    from scrapers.combined_scraper import run_combined_scraper
    return run_combined_scraper(city)

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
    from restaurant_rag import RestaurantRAG, demo_rag
    
    if data_file is None:
        data_file = get_latest_data_file()
        if not data_file:
            logger.error("No data file found. Please run the scraper first.")
            return
    
    demo_rag()

def run_streamlit():
    """Launch the Streamlit app."""
    import subprocess
    streamlit_path = os.path.join(os.path.dirname(__file__), "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_path])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restaurant Knowledge System")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Scraper command
    scraper_parser = subparsers.add_parser("scrape", help="Run the scraper")
    scraper_parser.add_argument("city", help="City to scrape (e.g., roorkee, ncr)")
    scraper_parser.add_argument("--force", action="store_true", help="Force re-scraping even if recent data exists")
    
    # Knowledge base command
    kb_parser = subparsers.add_parser("kb", help="Run knowledge base queries")
    kb_parser.add_argument("--data", help="Path to data file (will use latest if not specified)")
    
    # RAG demo command
    rag_parser = subparsers.add_parser("rag", help="Run RAG demo")
    rag_parser.add_argument("--data", help="Path to data file (will use latest if not specified)")
    rag_parser.add_argument("--full", action="store_true", help="Use full-size model instead of lightweight")
    
    # Streamlit command
    streamlit_parser = subparsers.add_parser("ui", help="Launch Streamlit UI")
    
    args = parser.parse_args()
    
    if args.command == "scrape":
        run_scraper(args.city, args.force)
    elif args.command == "kb":
        run_knowledge_base(args.data)
    elif args.command == "rag":
        run_rag_demo(args.data, not args.full)
    elif args.command == "ui":
        run_streamlit()
    else:
        parser.print_help()