"""
Restaurant Assistant UI - Compatibility Layer

This module redirects to the modular UI implementation in the ui package.
This is kept for backward compatibility with existing scripts or documentation.
"""

import logging
import os
import sys
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Show deprecation warning
warnings.warn(
    "streamlit_app.py is deprecated and will be removed in future versions. "
    "Please use 'python run.py ui' instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    logger.info("Redirecting to modular UI implementation...")
    
    # Get the path to the modular UI implementation
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app.py")
    
    # Check if file exists
    if not os.path.exists(ui_path):
        logger.error(f"UI implementation not found at {ui_path}")
        sys.exit(1)
    
    # Launch the modular UI
    import streamlit.web.bootstrap as bootstrap
    
    # Adjust sys.argv to point to the correct file
    sys.argv[0] = ui_path
    
    # Run the Streamlit app
    bootstrap.run(ui_path, "", [], {})