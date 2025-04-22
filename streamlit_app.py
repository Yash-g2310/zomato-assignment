import streamlit as st
import os
import sys
import json
import pandas as pd
import time
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import RAG system
from restaurant_rag import RestaurantRAG, find_latest_json_file

# Set page config
st.set_page_config(
    page_title="Restaurant Assistant",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #ebf5fb;
    }
    .chat-message.bot {
        background-color: #f0f8ef;
    }
    .chat-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
    }
    .chat-icon.user {
        background-color: #2874a6;
        color: white;
    }
    .chat-icon.bot {
        background-color: #27ae60;
        color: white;
    }
    .chat-content {
        flex: 1;
    }
    .menu-item {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 8px;
    }
    .veg {
        border-left: 4px solid #27ae60;
    }
    .non-veg {
        border-left: 4px solid #e74c3c;
    }
    .unknown {
        border-left: 4px solid #7f8c8d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loading" not in st.session_state:
    st.session_state.loading = False

def initialize_rag():
    """Initialize the RAG system."""
    with st.spinner("Setting up Restaurant Assistant..."):
        try:
            # Find the latest JSON file
            json_file = find_latest_json_file()
            
            # Initialize RAG system
            use_lightweight = st.session_state.get("use_lightweight", True)
            clear_cache = st.session_state.get("clear_cache", False)
            
            st.session_state.rag = RestaurantRAG(
                json_file, 
                use_lightweight=use_lightweight,
                clear_cache=clear_cache
            )
            
            st.success(f"Restaurant Assistant is ready with {len(st.session_state.rag.restaurant_docs)} restaurants!")
            return True
            
        except Exception as e:
            st.error(f"Error initializing Restaurant Assistant: {str(e)}")
            return False

def display_chat_message(message, is_user=False):
    """Display a chat message with styling."""
    message_type = "user" if is_user else "bot"
    icon = "👤" if is_user else "🍽️"
    
    with st.container():
        col1, col2 = st.columns([1, 12])
        
        with col1:
            st.markdown(f"""
            <div class="chat-icon {message_type}">
                {icon}
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="chat-content">
                {message}
            </div>
            """, unsafe_allow_html=True)

def process_restaurant_data(restaurant_info):
    """Process and display restaurant information."""
    if not restaurant_info:
        return "No restaurant information available."
    
    info = restaurant_info["info"]
    price = restaurant_info["price_range"]
    
    # Format restaurant information
    restaurant_html = f"""
    <h3>{restaurant_info['name']}</h3>
    <p><strong>Cuisine:</strong> {info['cuisine']}</p>
    <p><strong>Address:</strong> {info['address']}</p>
    <p><strong>Status:</strong> {info['status']}</p>
    <p><strong>Phone:</strong> {info['phone']}</p>
    <p><strong>Ratings:</strong> {json.dumps(info['ratings'])}</p>
    <p><strong>Price Range:</strong> ₹{price['min']} - ₹{price['max']} (avg ₹{price['avg']:.2f})</p>
    """
    
    if restaurant_info['sample_dishes']:
        restaurant_html += "<h4>Sample Menu Items:</h4>"
        for dish in restaurant_info['sample_dishes']:
            dish_type = dish.get('type', 'unknown')
            restaurant_html += f"""
            <div class="menu-item {dish_type}">
                <strong>{dish['name']}</strong> - {dish['price']} 
                <span style="font-size: 0.8em; color: #666;">({dish['category']})</span>
            </div>
            """
    
    return restaurant_html

def process_dish_results(dish_results):
    """Process and display dish search results."""
    if not dish_results:
        return "No dishes found matching your query."
    
    dishes_by_restaurant = {}
    
    # Group dishes by restaurant
    for dish in dish_results:
        restaurant = dish['restaurant']
        if restaurant not in dishes_by_restaurant:
            dishes_by_restaurant[restaurant] = []
        dishes_by_restaurant[restaurant].append(dish)
    
    # Create HTML for dish results
    dish_html = ""
    for restaurant, dishes in dishes_by_restaurant.items():
        dish_html += f"<h4>{restaurant}</h4>"
        for dish in dishes:
            dish_type = dish.get('type', 'unknown')
            dish_html += f"""
            <div class="menu-item {dish_type}">
                <strong>{dish['dish_name']}</strong> - {dish['price']} 
                <span style="font-size: 0.8em; color: #666;">({dish['category']})</span>
            </div>
            """
    
    return dish_html

# Sidebar
st.sidebar.title("Restaurant Assistant")
st.sidebar.image("https://img.icons8.com/cute-clipart/64/000000/restaurant.png")

# Settings
st.sidebar.subheader("Settings")
lightweight_option = st.sidebar.checkbox("Use Lightweight Model", value=True, 
                                       help="Enable for faster responses with less memory usage")
clear_cache_option = st.sidebar.checkbox("Clear Vector Cache", value=False,
                                       help="Enable to rebuild vector database (slower but may improve results)")

if st.sidebar.button("Initialize Assistant"):
    st.session_state.use_lightweight = lightweight_option
    st.session_state.clear_cache = clear_cache_option
    initialize_rag()

# Tools
st.sidebar.subheader("Tools")
if st.session_state.rag is not None:
    tool_option = st.sidebar.selectbox(
        "Select Tool",
        ["Chat Assistant", "Restaurant Search", "Dish Search"]
    )
else:
    tool_option = "Chat Assistant"
    st.sidebar.info("Initialize the assistant first to use tools.")

# Main content
st.title("Restaurant Knowledge Assistant")

if tool_option == "Chat Assistant":
    # Chat interface
    if st.session_state.rag is None:
        st.info("Please initialize the Restaurant Assistant using the sidebar button.")
    else:
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(message["content"], message["role"] == "user")
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Your question:", placeholder="Ask about restaurants or dishes...")
            submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Process with RAG
                with st.spinner("Thinking..."):
                    result = st.session_state.rag.query(user_input)
                    
                # Add assistant response to chat
                response = result["answer"]
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Force refresh to show new messages
                st.rerun()

elif tool_option == "Restaurant Search" and st.session_state.rag is not None:
    st.header("Restaurant Search")
    
    # Get all restaurant names
    restaurant_names = list(st.session_state.rag.kb.restaurants.keys())
    
    selected_restaurant = st.selectbox("Select a restaurant:", restaurant_names)
    
    if st.button("Show Restaurant Details"):
        with st.spinner("Fetching restaurant information..."):
            restaurant_info = st.session_state.rag.get_restaurant_info(selected_restaurant)
            
            if restaurant_info:
                st.markdown(process_restaurant_data(restaurant_info), unsafe_allow_html=True)
            else:
                st.error("Restaurant information not found.")

elif tool_option == "Dish Search" and st.session_state.rag is not None:
    st.header("Dish Search")
    
    dish_query = st.text_input("Search for a dish:", placeholder="e.g., paneer, pizza, chicken...")
    
    if st.button("Search Dishes"):
        if dish_query:
            with st.spinner("Searching for dishes..."):
                dish_results = st.session_state.rag.search_dishes(dish_query)
                
                if dish_results:
                    st.markdown(process_dish_results(dish_results), unsafe_allow_html=True)
                    
                    # Create a dataframe for easier viewing
                    df = pd.DataFrame(dish_results)
                    st.dataframe(df.sort_values(by=['restaurant', 'dish_name']),
                                use_container_width=True)
                else:
                    st.info(f"No dishes found matching '{dish_query}'")
        else:
            st.warning("Please enter a dish to search for.")

# Footer
st.markdown("---")
st.markdown("Restaurant Assistant powered by LangChain and Hugging Face models.")