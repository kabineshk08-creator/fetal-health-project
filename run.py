"""
Main entry point for Fetal Health Classification System
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import everything needed
from app_advanced import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)