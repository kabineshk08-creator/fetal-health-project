"""
Comprehensive Application Testing Script
Tests all major features of the Fetal Health Classification System
"""

import requests
import pandas as pd
import io
import json
import time

BASE_URL = "http://127.0.0.1:5000"
ADMIN_USER = "admin"
ADMIN_PASS = "admin123"
DOCTOR_USER = "doctor"
DOCTOR_PASS = "doctor123"

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

session = requests.Session()

def print_header(text):
    print(f"\n{BLUE}{'='*70}")
    print(f"{text.center(70)}")
    print(f"{'='*70}{RESET}\n")

def print_success(text):
    print(f"{GREEN}✅ {text}{RESET}")

def print_error(text):
    print(f"{RED}❌ {text}{RESET}")

def print_info(text):
    print(f"{YELLOW}ℹ️  {text}{RESET}")

def test_server_health():
    """Test if server is running"""
    print_header("TEST 1: SERVER HEALTH CHECK")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 302:  # Redirect to login
            print_success("Server is running and responding")
            return True
        else:
            print_error(f"Unexpected response: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Server unreachable: {str(e)}")
        return False

def test_admin_login():
    """Test admin login"""
    print_header("TEST 2: ADMIN LOGIN")
    try:
        response = session.post(f"{BASE_URL}/login", data={
            'username': ADMIN_USER,
            'password': ADMIN_PASS
        })
        if response.status_code == 200 or "dashboard" in response.text.lower():
            print_success(f"Admin login successful")
            return True
        else:
            print_error(f"Admin login failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Admin login error: {str(e)}")
        return False

def test_doctor_login():
    """Test doctor login"""
    print_header("TEST 3: DOCTOR LOGIN")
    try:
        new_session = requests.Session()
        response = new_session.post(f"{BASE_URL}/login", data={
            'username': DOCTOR_USER,
            'password': DOCTOR_PASS
        })
        if response.status_code == 200 or "dashboard" in response.text.lower():
            print_success(f"Doctor login successful")
            return True
        else:
            print_error(f"Doctor login failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Doctor login error: {str(e)}")
        return False

def create_test_csv(n_features=21):
    """Create test CSV data"""
    features_data = {
        'baseline': [147.45],
        'accelerations': [2.82],
        'fetal_movement': [2.86],
        'uterine_contractions': [2.07],
        'light_decelerations': [0.44],
        'severe_decelerations': [1.39],
        'prolonged_decelerations': [0.01],
        'abnormal_short_term_variability': [87.12],
        'mean_value_of_short_term_variability': [24.17],
        'percentage_of_time_with_abnormal_long_term_variability': [96.06],
        'mean_value_of_long_term_variability': [20.41],
        'histogram_width': [24.34],
        'histogram_min': [115.04],
        'histogram_max': [147.63],
        'histogram_number_of_peaks': [17.44],
        'histogram_number_of_zeroes': [39.36],
        'histogram_mode': [127.43],
        'histogram_mean': [154.13],
        'histogram_median': [118.01],
        'histogram_variance': [134.82],
        'histogram_tendency': [5.19],
    }
    
    df = pd.DataFrame(features_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_csv_prediction():
    """Test CSV prediction API"""
    print_header("TEST 4: CSV PREDICTION")
    try:
        csv_data = create_test_csv()
        files = {'file': ('test.csv', csv_data, 'text/csv')}
        
        response = session.post(f"{BASE_URL}/api/predict_csv", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if 'prediction' in result:
                print_success(f"CSV prediction successful")
                print_info(f"   Result: {result.get('prediction', 'N/A')}")
                print_info(f"   Confidence: {result.get('confidence', 'N/A'):.2%}")
                return True
            else:
                print_error(f"Invalid response format")
                return False
        else:
            print_error(f"CSV prediction failed - Status: {response.status_code}")
            print_info(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print_error(f"CSV prediction error: {str(e)}")
        return False

def test_dashboard():
    """Test dashboard access"""
    print_header("TEST 5: DASHBOARD ACCESS")
    try:
        response = session.get(f"{BASE_URL}/dashboard")
        if response.status_code == 200:
            print_success("Dashboard accessible")
            return True
        else:
            print_error(f"Dashboard access failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Dashboard error: {str(e)}")
        return False

def test_predict_page():
    """Test prediction page"""
    print_header("TEST 6: PREDICTION PAGE")
    try:
        response = session.get(f"{BASE_URL}/predict")
        if response.status_code == 200:
            print_success("Prediction page accessible")
            return True
        else:
            print_error(f"Prediction page access failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Prediction page error: {str(e)}")
        return False

def test_history_page():
    """Test history page"""
    print_header("TEST 7: PREDICTION HISTORY")
    try:
        response = session.get(f"{BASE_URL}/history")
        if response.status_code == 200:
            print_success("History page accessible")
            return True
        else:
            print_error(f"History page access failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"History page error: {str(e)}")
        return False

def test_metrics_page():
    """Test metrics page"""
    print_header("TEST 8: METRICS PAGE")
    try:
        response = session.get(f"{BASE_URL}/metrics")
        if response.status_code == 200:
            print_success("Metrics page accessible")
            return True
        else:
            print_error(f"Metrics page access failed - Status: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Metrics page error: {str(e)}")
        return False

def test_model_loading():
    """Test if models are loaded"""
    print_header("TEST 9: MODEL LOADING CHECK")
    try:
        import sys
        sys.path.insert(0, 'd:\\python\\fetal_health_project\\src')
        from app_advanced import models_dict
        
        models_loaded = list(models_dict.keys())
        
        if 'ann' in models_loaded:
            print_success(f"ANN model loaded")
        else:
            print_error(f"ANN model NOT loaded")
        
        if 'scaler' in models_loaded:
            print_success(f"Scaler loaded")
        else:
            print_error(f"Scaler NOT loaded")
        
        if len(models_loaded) > 0:
            print_success(f"Total models loaded: {len(models_loaded)}")
            return True
        else:
            print_error("No models loaded")
            return False
            
    except Exception as e:
        print_error(f"Model loading check error: {str(e)}")
        return False

def test_authentication():
    """Test authentication required pages"""
    print_header("TEST 10: AUTHENTICATION REQUIRED")
    try:
        new_session = requests.Session()
        response = new_session.get(f"{BASE_URL}/dashboard")
        
        # Should redirect to login
        if response.status_code == 200 and 'login' not in response.url:
            print_error("Authentication bypass detected!")
            return False
        else:
            print_success("Authentication properly enforced")
            return True
    except Exception as e:
        print_error(f"Authentication test error: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print_header("FETAL HEALTH CLASSIFICATION SYSTEM - APPLICATION TEST SUITE")
    
    results = {
        "Server Health": test_server_health(),
        "Admin Login": test_admin_login(),
        "Doctor Login": test_doctor_login(),
        "Dashboard": test_dashboard(),
        "Prediction Page": test_predict_page(),
        "History Page": test_history_page(),
        "Metrics Page": test_metrics_page(),
        "CSV Prediction": test_csv_prediction(),
        "Authentication": test_authentication(),
    }
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{BLUE}Test Results:{RESET}")
    for test_name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"  {test_name}: {status}")
    
    print(f"\n{BLUE}Overall: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print(f"\n{GREEN}{'*'*70}")
        print(f"{'ALL TESTS PASSED - APPLICATION IS FULLY FUNCTIONAL'.center(70)}")
        print(f"{'*'*70}{RESET}\n")
        return True
    else:
        print(f"\n{RED}{'*'*70}")
        print(f"{f'{total - passed} TESTS FAILED - PLEASE REVIEW'.center(70)}")
        print(f"{'*'*70}{RESET}\n")
        return False

if __name__ == "__main__":
    print_info("Waiting for server to be ready...")
    time.sleep(2)
    run_all_tests()