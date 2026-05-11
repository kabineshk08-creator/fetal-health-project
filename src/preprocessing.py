"""
Data Preprocessing Module for Fetal Health Classification
Handles: Missing values, scaling, feature selection, visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

class FetalHealthPreprocessor:
    def __init__(self, scaling_method='standard'):
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = None
        self.pca = None
        self.X_scaled = None
        self.feature_names = None
        self.selected_features = None
        self.smote = SMOTE(random_state=42)
        
    def apply_smote(self, X, y):
        """Apply SMOTE for class balancing"""
        print("Applying SMOTE for class balancing...")
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        print(f"✅ SMOTE applied - Original: {len(y)}, Resampled: {len(y_resampled)}")
        return X_resampled, y_resampled
    
    def extract_statistical_features(self, df):
        """Extract statistical features from CTG signals"""
        print("Extracting statistical features...")
        
        features = {}
        
        # Basic statistical features
        features['mean_fhr'] = df['baseline'].mean()
        features['std_fhr'] = df['baseline'].std()
        features['min_fhr'] = df['baseline'].min()
        features['max_fhr'] = df['baseline'].max()
        
        # Variability features
        features['fhr_variability'] = df['abnormal_short_term_variability'].mean()
        features['long_term_variability'] = df['mean_value_of_long_term_variability'].mean()
        
        # UC features
        features['uc_frequency'] = df['uterine_contractions'].mean()
        
        # Deceleration features
        features['decel_light'] = df['light_decelerations'].mean()
        features['decel_severe'] = df['severe_decelerations'].mean()
        features['decel_prolonged'] = df['prolonged_decelerations'].mean()
        
        # Acceleration features
        features['accel_count'] = df['accelerations'].mean()
        
        # Histogram features
        features['hist_mode'] = df['histogram_mode'].mean()
        features['hist_mean'] = df['histogram_mean'].mean()
        features['hist_median'] = df['histogram_median'].mean()
        features['hist_variance'] = df['histogram_variance'].mean()
        
        print("✅ Statistical features extracted")
        return pd.DataFrame([features])
    
    def extract_time_series_features(self, df):
        """Extract time-series features"""
        print("Extracting time-series features...")
        
        features = {}
        
        # Short-term variability (rolling window)
        if len(df) > 10:
            features['st_var_rolling'] = df['baseline'].rolling(window=10).std().mean()
        
        # Long-term trends
        features['fhr_trend'] = df['baseline'].diff().mean()
        
        # Peak analysis
        features['fhr_peaks'] = (df['baseline'] > df['baseline'].quantile(0.95)).sum()
        
        print("✅ Time-series features extracted")
        return pd.DataFrame([features]) if features else pd.DataFrame()
        
    def handle_missing_values(self, X):
        """Handle missing values using mean imputation"""
        print(f"Handling missing values...")
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns
        )
        print(f"✅ Missing values handled")
        return X_imputed
    
    def scale_features(self, X, fit=True):
        """Scale features using StandardScaler or MinMaxScaler"""
        print(f"Scaling features...")
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        print(f"✅ Features scaled")
        return X_scaled
    
    def select_best_features(self, X, y, n_features=15, method='f_classif'):
        """Select k-best features"""
        print(f"Selecting top {n_features} features using {method}...")
        
        score_func = f_classif if method == 'f_classif' else mutual_info_classif
        self.feature_selector = SelectKBest(score_func=score_func, k=n_features)
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = X.columns[selected_indices].tolist()
        
        print(f"✅ Selected features: {self.selected_features}")
        return pd.DataFrame(X_selected, columns=self.selected_features)
    
    def apply_pca(self, X, n_components=10):
        """Apply PCA for dimensionality reduction"""
        print(f"Applying PCA with {n_components} components...")
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        
        explained_variance = sum(self.pca.explained_variance_ratio_) * 100
        print(f"✅ PCA applied - Explained variance: {explained_variance:.2f}%")
        return X_pca
    
    def create_heatmap(self, X, filename='heatmap.png'):
        """Create correlation heatmap"""
        print(f"Creating correlation heatmap...")
        plt.figure(figsize=(12, 10))
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        correlation = X.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Heatmap saved to {filename}")
        return filename
    
    def create_distribution_plots(self, X, y, filename='distributions.png'):
        """Create distribution plots for features"""
        print(f"Creating distribution plots...")
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        
        n_features = min(9, X.shape[1])
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        class_names = ['Normal', 'Suspect', 'Pathological']
        colors = ['green', 'orange', 'red']
        
        for idx, col in enumerate(X.columns[:n_features]):
            for class_idx in range(3):
                mask = y == class_idx
                axes[idx].hist(X.loc[mask, col], alpha=0.6, label=class_names[class_idx], color=colors[class_idx], bins=20)
            
            axes[idx].set_title(f'Feature: {col}', fontsize=10, fontweight='bold')
            axes[idx].legend()
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Distribution plots saved to {filename}")
        return filename
    
    def preprocess_pipeline(self, X, y, n_features=15, apply_pca=False, apply_smote=True):
        """Complete preprocessing pipeline"""
        print("=" * 60)
        print("STARTING PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Select best features
        X = self.select_best_features(X, y, n_features=n_features)
        
        # Apply SMOTE if requested
        if apply_smote:
            X, y = self.apply_smote(X, y)
        
        # Scale features
        self.X_scaled = self.scale_features(X, fit=True)
        X_scaled = pd.DataFrame(self.X_scaled, columns=X.columns)
        
        # Apply PCA if requested
        if apply_pca:
            X_scaled = self.apply_pca(X_scaled, n_components=10)
        
        print("=" * 60)
        print("✅ PREPROCESSING PIPELINE COMPLETED")
        print("=" * 60)
        
        return X_scaled, y
