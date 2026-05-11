"""
Configuration settings for Fetal Health Classification Application
"""

import os
from datetime import timedelta

# Flask Configuration
SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
DEBUG = os.environ.get('FLASK_DEBUG', False)
TESTING = False

# Database Configuration
SQLALCHEMY_DATABASE_URI = 'sqlite:///app.db'
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Session Configuration
PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
SESSION_COOKIE_SECURE = False  # Set to True in production with HTTPS
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'

# File Upload Configuration
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'jpg', 'jpeg', 'png', 'gif'}

# Model Configuration
MODEL_PATHS = {
    'ann': '../models/ann_model.h5',
    'cnn': '../models/cnn_model.h5',
    'scaler': '../models/scaler.pkl',
}

# Logging Configuration
LOG_FILE = 'logs/app.log'
LOG_LEVEL = 'INFO'

# Application Settings
APP_NAME = 'Fetal Health Classification System'
APP_VERSION = '1.0.0'
ORGANIZATION = 'Healthcare AI Solutions'

# Model Configuration
INPUT_SHAPE_ANN = (21,)  # 21 features for ANN
INPUT_SHAPE_CNN = (128, 96, 1)  # Image shape for CNN
NUM_CLASSES = 3
CLASS_NAMES = {1: 'Normal', 2: 'Suspect', 3: 'Pathological'}

# Prediction Confidence Threshold
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence for prediction

# Data Preprocessing Configuration
FEATURE_SCALING = 'standard'  # 'standard' or 'minmax'
PCA_COMPONENTS = 10
NUM_TOP_FEATURES = 15

# Training Configuration
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2

print("✅ Configuration loaded successfully!")
