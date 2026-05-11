#!/usr/bin/env python
"""
Create sample CTG (Cardiotocography) data for training
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create synthetic CTG data"""
    print("\n" + "=" * 70)
    print("CREATING SAMPLE CTG DATA")
    print("=" * 70)
    
    np.random.seed(42)
    n_samples = 300
    
    print(f"\nGenerating {n_samples} samples...")
    
    # Create features
    features = {
        'baseline': np.random.normal(140, 15, n_samples),
        'accelerations': np.random.uniform(0, 10, n_samples),
        'fetal_movement': np.random.uniform(0, 5, n_samples),
        'uterine_contractions': np.random.uniform(0, 3, n_samples),
        'light_decelerations': np.random.uniform(0, 5, n_samples),
        'severe_decelerations': np.random.uniform(0, 2, n_samples),
        'prolonged_decelerations': np.random.uniform(0, 1, n_samples),
        'abnormal_short_term_variability': np.random.uniform(0, 100, n_samples),
        'mean_value_of_short_term_variability': np.random.uniform(0, 50, n_samples),
        'percentage_of_time_with_abnormal_long_term_variability': np.random.uniform(0, 100, n_samples),
        'mean_value_of_long_term_variability': np.random.uniform(0, 50, n_samples),
        'histogram_width': np.random.uniform(0, 100, n_samples),
        'histogram_min': np.random.uniform(50, 120, n_samples),
        'histogram_max': np.random.uniform(140, 220, n_samples),
        'histogram_number_of_peaks': np.random.uniform(0, 20, n_samples),
        'histogram_number_of_zeroes': np.random.uniform(0, 100, n_samples),
        'histogram_mode': np.random.uniform(100, 180, n_samples),
        'histogram_mean': np.random.uniform(100, 180, n_samples),
        'histogram_median': np.random.uniform(100, 180, n_samples),
        'histogram_variance': np.random.uniform(0, 500, n_samples),
        'histogram_tendency': np.random.uniform(-10, 10, n_samples),
    }
    
    # Create target variable
    fetal_health = np.random.choice([1, 2, 3], n_samples, p=[0.78, 0.17, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame(features)
    df['fetal_health'] = fetal_health
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv('data/sample_ctg.csv', index=False)
    
    print(f"\n✅ Sample data created successfully!")
    print(f"   📁 File: data/sample_ctg.csv")
    print(f"   📊 Samples: {len(df)}")
    print(f"   📋 Features: {len(df.columns) - 1}")
    print(f"\n   Class Distribution:")
    print(f"   - Normal (1): {(fetal_health == 1).sum()} ({(fetal_health == 1).sum() / len(df) * 100:.1f}%)")
    print(f"   - Suspect (2): {(fetal_health == 2).sum()} ({(fetal_health == 2).sum() / len(df) * 100:.1f}%)")
    print(f"   - Pathological (3): {(fetal_health == 3).sum()} ({(fetal_health == 3).sum() / len(df) * 100:.1f}%)")
    
    return df

if __name__ == '__main__':
    create_sample_data()
