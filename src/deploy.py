"""
Deployment Module for Fetal Health Classification
Real-time prediction script with alerts
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import time
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FetalHealthDeployment:
    def __init__(self, model_path='../models/', scaler_path='../models/scaler.pkl'):
        self.models = {}
        self.scaler = None
        self.load_models(model_path, scaler_path)
        
    def load_models(self, model_path, scaler_path):
        """Load trained models"""
        try:
            # Load ANN model
            if tf.io.gfile.exists(f'{model_path}ann_model.h5'):
                self.models['ann'] = tf.keras.models.load_model(f'{model_path}ann_model.h5')
                logger.info("ANN model loaded")
            
            # Load CNN model
            if tf.io.gfile.exists(f'{model_path}cnn_model.h5'):
                self.models['cnn'] = tf.keras.models.load_model(f'{model_path}cnn_model.h5')
                logger.info("CNN model loaded")
            
            # Load scaler
            if tf.io.gfile.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded")
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def preprocess_realtime_data(self, ctg_data):
        """Preprocess real-time CTG data"""
        try:
            # Convert to DataFrame if dict
            if isinstance(ctg_data, dict):
                df = pd.DataFrame([ctg_data])
            else:
                df = ctg_data
            
            # Remove target column if present
            if 'fetal_health' in df.columns:
                df = df.drop('fetal_health', axis=1)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(df.values)
            else:
                features_scaled = df.values
            
            return features_scaled
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return None
    
    def predict_realtime(self, ctg_data, model_type='ann'):
        """Make real-time prediction"""
        try:
            # Preprocess data
            features = self.preprocess_realtime_data(ctg_data)
            if features is None:
                return None
            
            # Select model
            if model_type not in self.models:
                logger.error(f"Model {model_type} not available")
                return None
            
            model = self.models[model_type]
            
            # Make prediction
            if model_type in ['cnn']:
                # Reshape for CNN
                features = features.reshape(features.shape[0], features.shape[1], 1)
            
            predictions = model.predict(features, verbose=0)
            
            # Get results
            class_idx = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            
            class_names = ['Normal', 'Suspect', 'Pathological']
            result = class_names[class_idx]
            
            prediction_result = {
                'prediction': result,
                'confidence': confidence,
                'probabilities': {
                    'Normal': float(predictions[0][0]),
                    'Suspect': float(predictions[0][1]),
                    'Pathological': float(predictions[0][2])
                },
                'timestamp': datetime.now().isoformat(),
                'model_used': model_type
            }
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def generate_alert(self, prediction_result, threshold=0.7):
        """Generate alerts based on prediction"""
        alerts = []
        
        if prediction_result['prediction'] == 'Pathological':
            alerts.append({
                'level': 'CRITICAL',
                'message': '🚨 CRITICAL: Pathological fetal health detected!',
                'action': 'Immediate medical intervention required'
            })
        
        elif prediction_result['prediction'] == 'Suspect':
            alerts.append({
                'level': 'WARNING',
                'message': '⚠️ WARNING: Suspect fetal health pattern detected',
                'action': 'Monitor closely and consult specialist'
            })
        
        if prediction_result['confidence'] < threshold:
            alerts.append({
                'level': 'INFO',
                'message': f'ℹ️ Low confidence prediction ({prediction_result["confidence"]:.2f})',
                'action': 'Consider additional tests'
            })
        
        return alerts
    
    def stream_predictions(self, data_stream, model_type='ann', interval=5):
        """Process streaming CTG data"""
        logger.info("Starting real-time prediction stream...")
        
        for i, ctg_data in enumerate(data_stream):
            logger.info(f"Processing data point {i+1}")
            
            # Make prediction
            result = self.predict_realtime(ctg_data, model_type)
            
            if result:
                # Generate alerts
                alerts = self.generate_alert(result)
                
                # Log results
                logger.info(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.2f})")
                
                # Print alerts
                for alert in alerts:
                    logger.warning(f"{alert['level']}: {alert['message']}")
                    logger.info(f"Action: {alert['action']}")
            
            # Wait for next data point
            time.sleep(interval)
    
    def simulate_realtime_monitoring(self, sample_data_path='data/sample_ctg.csv', model_type='ann'):
        """Simulate real-time monitoring with sample data"""
        try:
            # Load sample data
            df = pd.read_csv(sample_data_path)
            
            # Remove target if present
            if 'fetal_health' in df.columns:
                df = df.drop('fetal_health', axis=1)
            
            # Simulate streaming
            logger.info("Starting simulation of real-time fetal health monitoring...")
            
            for idx, row in df.iterrows():
                ctg_data = row.to_dict()
                
                # Make prediction
                result = self.predict_realtime(ctg_data, model_type)
                
                if result:
                    alerts = self.generate_alert(result)
                    
                    print(f"\n{'='*60}")
                    print(f"CTG Sample {idx+1}")
                    print(f"{'='*60}")
                    print(f"Prediction: {result['prediction']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    
                    if alerts:
                        print("ALERTS:")
                        for alert in alerts:
                            print(f"  {alert['level']}: {alert['message']}")
                            print(f"  Action: {alert['action']}")
                    
                    print(f"Probabilities: Normal={result['probabilities']['Normal']:.3f}, "
                          f"Suspect={result['probabilities']['Suspect']:.3f}, "
                          f"Pathological={result['probabilities']['Pathological']:.3f}")
                
                # Simulate real-time delay
                time.sleep(2)
                
                # Stop after 10 samples for demo
                if idx >= 9:
                    break
            
            logger.info("Simulation completed")
            
        except Exception as e:
            logger.error(f"Simulation error: {e}")

if __name__ == "__main__":
    # Example usage
    deployment = FetalHealthDeployment()
    
    # Simulate real-time monitoring
    deployment.simulate_realtime_monitoring()