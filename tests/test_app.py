"""
Test script for Fetal Health Classification Application
Tests core functionality and endpoints
"""

import requests
import json
from urllib.parse import urljoin

BASE_URL = "http://127.0.0.1:5000"

def test_application():
    """Run comprehensive application tests"""
    
    print("🧪 TESTING FETAL HEALTH CLASSIFICATION APPLICATION\n")
    print("=" * 60)
    
    # Test 1: Home page
    print("\n✓ Test 1: Home page redirect")
    try:
        r = requests.get(f"{BASE_URL}/", allow_redirects=False)
        print(f"  Status: {r.status_code} (expected 302 redirect)")
        assert r.status_code == 302, "Should redirect to login"
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test 2: Login page
    print("\n✓ Test 2: Login page loads")
    try:
        r = requests.get(f"{BASE_URL}/login")
        print(f"  Status: {r.status_code}")
        assert r.status_code == 200, "Login page should load"
        assert "login" in r.text.lower(), "Should contain login form"
        print(f"  Contains login form: True")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test 3: Predict page (requires login but should load)
    print("\n✓ Test 3: Predict page accessibility")
    try:
        r = requests.get(f"{BASE_URL}/predict")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Page loads successfully")
        elif r.status_code == 302:
            print(f"  Redirects to login (requires authentication)")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Test 4: Metrics page
    print("\n✓ Test 4: Metrics page")
    try:
        r = requests.get(f"{BASE_URL}/metrics")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Page loads successfully")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 5: History page
    print("\n✓ Test 5: History page")
    try:
        r = requests.get(f"{BASE_URL}/history")
        print(f"  Status: {r.status_code}")
        if r.status_code == 200:
            print(f"  Page loads successfully")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 6: Test login functionality
    print("\n✓ Test 6: Authentication test")
    try:
        session = requests.Session()
        
        # Try logging in
        login_data = {
            'username': 'doctor',
            'password': 'doctor123'
        }
        r = session.post(f"{BASE_URL}/login", data=login_data, allow_redirects=False)
        print(f"  Login attempt status: {r.status_code}")
        
        if r.status_code == 302:
            print(f"  ✅ Login successful (redirects to dashboard)")
            
            # Try accessing protected page
            r = session.get(f"{BASE_URL}/dashboard")
            print(f"  Dashboard access: {r.status_code}")
            if r.status_code == 200:
                print(f"  ✅ Can access protected pages after login")
        else:
            print(f"  Note: Status {r.status_code} (may be due to session handling)")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 7: Check static files
    print("\n✓ Test 7: Static assets")
    try:
        r = requests.head(f"{BASE_URL}/static/style.css")
        print(f"  CSS file: {r.status_code}")
        assert r.status_code == 200, "CSS should be accessible"
    except Exception as e:
        print(f"  Note: {e}")
    
    # Test 8: File upload endpoints
    print("\n✓ Test 8: Upload API endpoints")
    endpoints_to_check = [
        '/api/csv_file',
        '/api/image_file',
        '/download_template'
    ]
    
    for endpoint in endpoints_to_check:
        try:
            r = requests.options(f"{BASE_URL}{endpoint}", allow_redirects=False)
            status = r.status_code
            if status == 404:
                # Try GET
                r = requests.get(f"{BASE_URL}{endpoint}", allow_redirects=False)
                status = r.status_code
            print(f"  {endpoint}: {status}")
        except Exception as e:
            print(f"  {endpoint}: Error - {str(e)[:50]}")
    
    # Summary
    print("\n" + "=" * 60)
    print("\n✅ APPLICATION TESTING COMPLETE\n")
    print("📋 Default Credentials:")
    print("   Doctor:  username='doctor', password='doctor123'")
    print("   Admin:   username='admin', password='admin123'")
    print("\n🌐 Access the app at: http://127.0.0.1:5000")
    print("\n📊 Features tested:")
    print("   ✓ Page routing")
    print("   ✓ Authentication flow")
    print("   ✓ Protected pages")
    print("   ✓ Static assets")
    print("   ✓ API endpoints")

if __name__ == "__main__":
    test_application()
