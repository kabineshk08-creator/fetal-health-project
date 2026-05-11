"""
ML Models Module for Fetal Health Classification
Includes CNN, RNN, and Hybrid Ensemble models
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, Flatten, Input, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings

warnings.filterwarnings('ignore')

class FetalHealthModels:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        self.cnn_model = None
        self.rnn_model = None
        self.hybrid_model = None
        self.gb_model = None
        
    def build_cnn_model(self, input_shape):
        """Build CNN model for CTG signals"""
        print("Building CNN model...")
        
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.cnn_model = model
        print("✅ CNN model built")
        return model
    
    def build_rnn_model(self, input_shape):
        """Build RNN (LSTM) model for CTG signals"""
        print("Building RNN model...")
        
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            LSTM(32),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.rnn_model = model
        print("✅ RNN model built")
        return model
    
    def build_hybrid_ensemble(self, input_shape):
        """Build Hybrid Ensemble: CNN/RNN + Gradient Boosting"""
        print("Building Hybrid Ensemble model...")
        
        # CNN branch
        cnn_input = Input(shape=input_shape)
        cnn_branch = Conv1D(32, kernel_size=3, activation='relu')(cnn_input)
        cnn_branch = MaxPooling1D(pool_size=2)(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)
        cnn_branch = Dense(32, activation='relu')(cnn_branch)
        
        # RNN branch
        rnn_branch = LSTM(32, input_shape=input_shape)(cnn_input)
        rnn_branch = Dense(32, activation='relu')(rnn_branch)
        
        # Combine branches
        combined = concatenate([cnn_branch, rnn_branch])
        combined = Dense(64, activation='relu')(combined)
        combined = Dropout(0.5)(combined)
        output = Dense(3, activation='softmax')(combined)
        
        model = Model(inputs=cnn_input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        
        self.hybrid_model = model
        
        # Gradient Boosting for tabular features
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        print("✅ Hybrid Ensemble model built")
        return model
    
    def train_cnn(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train CNN model"""
        print("Training CNN model...")
        
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None and X_val.ndim == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("✅ CNN model trained")
        return history
    
    def train_rnn(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train RNN model"""
        print("Training RNN model...")
        
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None and X_val.ndim == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        history = self.rnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("✅ RNN model trained")
        return history
    
    def train_hybrid(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """Train Hybrid model"""
        print("Training Hybrid model...")
        
        if X_train.ndim == 2:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        if X_val is not None and X_val.ndim == 2:
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        history = self.hybrid_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        print("✅ Hybrid model trained")
        return history
    
    def predict_ensemble(self, X_cnn, X_gb, weights=None):
        """Make predictions using weighted soft voting"""
        print("Making ensemble predictions...")
        
        if X_cnn.ndim == 2:
            X_cnn = X_cnn.reshape(X_cnn.shape[0], X_cnn.shape[1], 1)
        
        # CNN/RNN predictions
        cnn_pred = self.hybrid_model.predict(X_cnn, verbose=0)
        
        # GB predictions
        gb_pred = self.gb_model.predict_proba(X_gb)
        
        # Weighted voting
        if weights is None:
            weights = [0.6, 0.4]  # Default weights
        
        ensemble_pred = weights[0] * cnn_pred + weights[1] * gb_pred
        
        final_pred = np.argmax(ensemble_pred, axis=1)
        confidence = np.max(ensemble_pred, axis=1)
        
        print("✅ Ensemble predictions completed")
        return final_pred, confidence, ensemble_pred
    
    def save_models(self, path='../models/'):
        """Save trained models"""
        if self.cnn_model:
            self.cnn_model.save(f'{path}cnn_model.h5')
        if self.rnn_model:
            self.rnn_model.save(f'{path}rnn_model.h5')
        if self.hybrid_model:
            self.hybrid_model.save(f'{path}hybrid_model.h5')
        if self.gb_model:
            joblib.dump(self.gb_model, f'{path}gb_model.pkl')
        print("✅ Models saved")
    
    def load_models(self, path='../models/'):
        """Load trained models"""
        try:
            if tf.io.gfile.exists(f'{path}cnn_model.h5'):
                self.cnn_model = tf.keras.models.load_model(f'{path}cnn_model.h5')
            if tf.io.gfile.exists(f'{path}rnn_model.h5'):
                self.rnn_model = tf.keras.models.load_model(f'{path}rnn_model.h5')
            if tf.io.gfile.exists(f'{path}hybrid_model.h5'):
                self.hybrid_model = tf.keras.models.load_model(f'{path}hybrid_model.h5')
            if tf.io.gfile.exists(f'{path}gb_model.pkl'):
                self.gb_model = joblib.load(f'{path}gb_model.pkl')
            print("✅ Models loaded")
        except Exception as e:
            print(f"❌ Error loading models: {e}")