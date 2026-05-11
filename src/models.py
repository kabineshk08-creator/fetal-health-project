"""
Database Models for Fetal Health Classification System
Includes User authentication and Prediction history
"""

import os
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='doctor')  # doctor, admin
    full_name = db.Column(db.String(120), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships - specify foreign_keys to avoid ambiguity with verified_by_user
    predictions = db.relationship('Prediction', 
                                 foreign_keys='Prediction.user_id',
                                 backref='user', 
                                 lazy=True, 
                                 cascade='all, delete-orphan')
    verified_predictions = db.relationship('Prediction',
                                          foreign_keys='Prediction.verified_by_user',
                                          backref='verifier',
                                          lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password"""
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    """Model to store prediction history"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    prediction_type = db.Column(db.String(20), nullable=False)  # csv, image
    result = db.Column(db.String(30), nullable=False)  # Normal, Suspect, Pathological
    confidence = db.Column(db.Float, nullable=False)
    probabilities = db.Column(db.String(500), nullable=True)  # JSON string
    file_name = db.Column(db.String(255), nullable=True)
    notes = db.Column(db.Text, nullable=True)
    # Accuracy tracking fields
    actual_label = db.Column(db.String(30), nullable=True)  # Ground truth label (Normal, Suspect, Pathological)
    is_correct = db.Column(db.Boolean, nullable=True)  # Whether prediction matches actual label
    accuracy_verified = db.Column(db.Boolean, default=False)  # Whether ground truth has been verified
    verified_by_user = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # Who verified
    verified_at = db.Column(db.DateTime, nullable=True)  # When ground truth was added
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    def set_actual_label(self, label):
        """Set actual label and calculate correctness"""
        self.actual_label = label
        self.is_correct = (self.result == label)
        self.accuracy_verified = True
        self.verified_at = datetime.utcnow()
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.result} (Correct: {self.is_correct})'

class ModelMetrics(db.Model):
    """Store model performance metrics"""
    __tablename__ = 'model_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(50), nullable=False)  # ann, cnn
    accuracy = db.Column(db.Float, nullable=False)
    precision = db.Column(db.Float, nullable=False)
    recall = db.Column(db.Float, nullable=False)
    f1_score = db.Column(db.Float, nullable=False)
    confusion_matrix = db.Column(db.String(500), nullable=True)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelMetrics {self.model_name}: {self.accuracy}>'

def init_db(app):
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")

def create_default_users(app):
    """Create default admin and doctor users"""
    with app.app_context():
        if User.query.filter_by(username='admin').first():
            return
        
        # Create admin user
        admin = User(
            username='admin',
            email='admin@fetalhealth.com',
            full_name='System Administrator',
            role='admin'
        )
        admin.set_password('admin123')
        
        # Create doctor user
        doctor = User(
            username='doctor',
            email='doctor@fetalhealth.com',
            full_name='Dr. Medical Professional',
            role='doctor'
        )
        doctor.set_password('doctor123')
        
        db.session.add(admin)
        db.session.add(doctor)
        db.session.commit()
        
        print("✅ Default users created")
        print("   Admin: username='admin', password='admin123'")
        print("   Doctor: username='doctor', password='doctor123'")
