"""
Integration test for CSV prediction with feature mismatch fix
Simulates actual file upload and prediction
"""

import requests
import pandas as pd
import io
import json

BASE_URL = "http://127.0.0.1:5000"

def create_test_csv_with_target():
    """Create a test CSV with 22 columns (21 features + target)"""
    
    features_data = {
        'baseline': [147.4507122951685],
        'accelerations': [2.820345725713065],
        'fetal_movement': [2.8580634473494992],
        'uterine_contractions': [2.0652071892832082],
        'light_decelerations': [0.43933249572416677],
        'severe_decelerations': [1.3902891080951876],
        'prolonged_decelerations': [0.009331619827093296],
        'abnormal_short_term_variability': [87.1204482263052],
        'mean_value_of_short_term_variability': [24.17037179865733],
        'percentage_of_time_with_abnormal_long_term_variability': [96.06269755131997],
        'mean_value_of_long_term_variability': [20.407584543334906],
        'histogram_width': [24.339668321853047],
        'histogram_min': [115.04013020481811],
        'histogram_max': [147.6288737018403],
        'histogram_number_of_peaks': [17.43826122561011],
        'histogram_number_of_zeroes': [39.355533610631554],
        'histogram_mode': [127.43437302073568],
        'histogram_mean': [154.1293786842005],
        'histogram_median': [118.01288193727456],
        'histogram_variance': [134.8150043835914],
        'histogram_tendency': [5.19040297433888],
        'fetal_health': [1],
    }
    
    df = pd.DataFrame(features_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def create_test_csv_without_target():
    """Create a test CSV with 21 columns (21 features, no target)"""
    
    features_data = {
        'baseline': [137.92603548243224],
        'accelerations': [1.7743954377972282],
        'fetal_movement': [1.3998954683014209],
        'uterine_contractions': [1.3044820158112387],
        'light_decelerations': [1.7531346560459038],
        'severe_decelerations': [0.8086375362085843],
        'prolonged_decelerations': [0.24667886452162302],
        'abnormal_short_term_variability': [76.60647765353026],
        'mean_value_of_short_term_variability': [43.17737612572726],
        'percentage_of_time_with_abnormal_long_term_variability': [19.692570445342838],
        'mean_value_of_long_term_variability': [0.7697445845654238],
        'histogram_width': [55.73371695730081],
        'histogram_min': [77.11257625827199],
        'histogram_max': [176.00469115409803],
        'histogram_number_of_peaks': [11.539632248353637],
        'histogram_number_of_zeroes': [13.07173235897573],
        'histogram_mode': [154.70960999272256],
        'histogram_mean': [129.07231455995415],
        'histogram_median': [152.98543308316658],
        'histogram_variance': [409.8385013902295],
        'histogram_tendency': [-5.880236316230536],
    }
    
    df = pd.DataFrame(features_data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def test_csv_prediction_with_target():
    """Test CSV prediction with the target column"""
    
    print("=" * 70)
    print("TEST 1: CSV PREDICTION WITH TARGET COLUMN (22 columns)")
    print("=" * 70)
    
    print("\n[1] Creating test CSV with 22 columns...")
    csv_data = create_test_csv_with_target()
    
    try:
        print("[2] Sending to Flask app for prediction...")
        files = {'file': ('test.csv', csv_data, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/predict_csv", files=files, timeout=10)
        
        print(f"    Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\n    ✅ PREDICTION SUCCESSFUL!")
                print(f"    Result: {result.get('prediction', 'N/A')}")
                print(f"    Confidence: {result.get('confidence', 'N/A'):.2f}%")
                return True
            except Exception as json_e:
                print(f"\n    ❌ JSON PARSE ERROR: {str(json_e)}")
                print(f"    Response content: {response.text[:500]}")
                return False
        else:
            print(f"\n    ⚠️  Unexpected status: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"\n    ❌ ERROR: {str(e)[:100]}")
        return False

def test_csv_prediction_without_target():
    """Test CSV prediction without the target column"""
    
    print("\n" + "=" * 70)
    print("TEST 2: CSV PREDICTION WITHOUT TARGET COLUMN (21 columns)")
    print("=" * 70)
    
    print("\n[1] Creating test CSV with 21 columns...")
    csv_data = create_test_csv_without_target()
    
    try:
        print("[2] Sending to Flask app for prediction...")
        files = {'file': ('test.csv', csv_data, 'text/csv')}
        response = requests.post(f"{BASE_URL}/api/predict_csv", files=files, timeout=10)
        
        print(f"    Response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\n    ✅ PREDICTION SUCCESSFUL!")
                print(f"    Result: {result.get('prediction', 'N/A')}")
                print(f"    Confidence: {result.get('confidence', 'N/A'):.2f}%")
                return True
            except Exception as json_e:
                print(f"\n    ❌ JSON PARSE ERROR: {str(json_e)}")
                print(f"    Response content: {response.text[:500]}")
                return False
        else:
            print(f"\n    ⚠️  Unexpected status: {response.status_code}")
            print(f"    Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"\n    ❌ ERROR: {str(e)[:100]}")
        return False

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  INTEGRATION TEST: CSV PREDICTION WITH FEATURE MISMATCH FIX".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    test1 = test_csv_prediction_with_target()
    test2 = test_csv_prediction_without_target()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY:")
    print("=" * 70)
    
    if test1:
        print("\n  ✅ Test 1 PASSED: CSV with target column (22 features)")
    else:
        print("\n  ❌ Test 1 FAILED: CSV with target column")
    
    if test2:
        print("  ✅ Test 2 PASSED: CSV without target column (21 features)")
    else:
        print("  ❌ Test 2 FAILED: CSV without target column")
    
    print("\n" + "=" * 70)
    
    if test1 and test2:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        print("\nThe feature mismatch issue is completely resolved.")
        print("CSV predictions are working correctly with both file formats.")
    else:
        print("\n⚠️  Some tests failed - review the output above")
    
    print("\n" + "=" * 70)
