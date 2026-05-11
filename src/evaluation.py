"""
Evaluation Module for Fetal Health Classification
Includes SHAP/LIME explainability and model evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import shap
import lime
import lime.lime_tabular
import warnings

warnings.filterwarnings('ignore')

class FetalHealthEvaluator:
    def __init__(self):
        self.shap_explainer = None
        self.lime_explainer = None
        
    def evaluate_model(self, y_true, y_pred, model_name="Model"):
        """Comprehensive model evaluation"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Normal', 'Suspect', 'Pathological']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
    
    def plot_confusion_matrix(self, cm, model_name, filename=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Suspect', 'Pathological'],
                   yticklabels=['Normal', 'Suspect', 'Pathological'])
        plt.title(f'Confusion Matrix - {model_name}', fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if filename:
            plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.show()
    
    def setup_shap_explainer(self, model, X_background):
        """Setup SHAP explainer"""
        print("Setting up SHAP explainer...")
        try:
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(model.predict_proba, X_background)
            else:
                self.shap_explainer = shap.Explainer(model, X_background)
            print("✅ SHAP explainer ready")
        except Exception as e:
            print(f"❌ SHAP setup failed: {e}")
    
    def explain_with_shap(self, X_instance, model_name="Model"):
        """Generate SHAP explanations"""
        if self.shap_explainer is None:
            print("❌ SHAP explainer not initialized")
            return None
        
        print(f"Generating SHAP explanation for {model_name}...")
        
        try:
            shap_values = self.shap_explainer(X_instance)
            
            # Plot summary
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_instance, show=False)
            plt.title(f'SHAP Summary Plot - {model_name}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'shap_summary_{model_name.lower()}.png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for first instance
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.title(f'SHAP Waterfall Plot - {model_name}', fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'shap_waterfall_{model_name.lower()}.png', dpi=100, bbox_inches='tight')
            plt.close()
            
            print("✅ SHAP explanations generated")
            return shap_values
            
        except Exception as e:
            print(f"❌ SHAP explanation failed: {e}")
            return None
    
    def setup_lime_explainer(self, X_train, feature_names=None, class_names=['Normal', 'Suspect', 'Pathological']):
        """Setup LIME explainer"""
        print("Setting up LIME explainer...")
        try:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train.values if hasattr(X_train, 'values') else X_train,
                feature_names=feature_names or [f'Feature_{i}' for i in range(X_train.shape[1])],
                class_names=class_names,
                mode='classification'
            )
            print("✅ LIME explainer ready")
        except Exception as e:
            print(f"❌ LIME setup failed: {e}")
    
    def explain_with_lime(self, model, X_instance, num_features=10):
        """Generate LIME explanations"""
        if self.lime_explainer is None:
            print("❌ LIME explainer not initialized")
            return None
        
        print("Generating LIME explanation...")
        
        try:
            # Get prediction function
            predict_fn = model.predict_proba if hasattr(model, 'predict_proba') else model.predict
            
            # Explain instance
            exp = self.lime_explainer.explain_instance(
                X_instance.values[0] if hasattr(X_instance, 'values') else X_instance[0],
                predict_fn,
                num_features=num_features
            )
            
            # Save explanation plot
            fig = exp.as_pyplot_figure()
            fig.suptitle('LIME Explanation', fontweight='bold')
            plt.tight_layout()
            plt.savefig('lime_explanation.png', dpi=100, bbox_inches='tight')
            plt.close()
            
            print("✅ LIME explanation generated")
            return exp
            
        except Exception as e:
            print(f"❌ LIME explanation failed: {e}")
            return None
    
    def get_top_features(self, shap_values, feature_names=None, top_n=10):
        """Extract top contributing features from SHAP"""
        if shap_values is None:
            return None
        
        try:
            # Get mean absolute SHAP values
            mean_shap = np.abs(shap_values.values).mean(axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_shap)[-top_n:][::-1]
            
            feature_importance = []
            for idx in top_indices:
                feature_name = feature_names[idx] if feature_names else f'Feature_{idx}'
                importance = mean_shap[idx]
                feature_importance.append((feature_name, importance))
            
            print("Top contributing features:")
            for name, imp in feature_importance:
                print(f"  {name}: {imp:.4f}")
            
            return feature_importance
            
        except Exception as e:
            print(f"❌ Feature extraction failed: {e}")
            return None