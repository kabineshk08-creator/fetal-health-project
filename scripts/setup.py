#!/usr/bin/env python
"""
Setup script for Fetal Health Classification System
Initializes the project structure and database
"""

import os
import sys
import subprocess

def setup_project():
    """Setup project"""
    print("=" * 70)
    print("FETAL HEALTH CLASSIFICATION SYSTEM - SETUP")
    print("=" * 70)
    
    # Create necessary directories
    print("\n1️⃣  Creating directories...")
    directories = [
        'models',
        'data',
        'data/images',
        'data/images/train',
        'data/images/test',
        'uploads',
        'static/uploads',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    # Install dependencies
    print("\n2️⃣  Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("   ✅ Dependencies installed")
    except Exception as e:
        print(f"   ⚠️  Error installing dependencies: {e}")
        return False
    
    # Create sample data
    print("\n3️⃣  Creating sample data...")
    try:
        exec(open('create_sample_data.py').read())
        print("   ✅ Sample data created")
    except Exception as e:
        print(f"   ⚠️  Error creating sample data: {e}")
    
    # Train models
    print("\n4️⃣  Training models...")
    print("   ⏳ This may take several minutes...")
    try:
        subprocess.check_call([sys.executable, 'train_advanced.py'])
        print("   ✅ Models trained successfully")
    except Exception as e:
        print(f"   ⚠️  Error training models: {e}")
        print("   You can train models manually by running: python train_advanced.py")
    
    # Initialize database
    print("\n5️⃣  Initializing database...")
    try:
        from app_advanced import app, db
        with app.app_context():
            db.create_all()
            from models import create_default_users
            create_default_users(app)
            print("   ✅ Database initialized with default users")
    except Exception as e:
        print(f"   ⚠️  Error initializing database: {e}")
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n📝 Next Steps:")
    print("   1. Run: python app_advanced.py")
    print("   2. Open: http://localhost:5000")
    print("   3. Login with:")
    print("      - Admin: admin / admin123")
    print("      - Doctor: doctor / doctor123")
    print("\n")

if __name__ == '__main__':
    setup_project()
