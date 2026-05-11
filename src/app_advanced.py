"""
Complete Flask Application for Fetal Health Classification
Includes: User authentication, predictions, history, metrics
"""

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from PIL import Image
import joblib
import json
from datetime import datetime
import io
from functools import wraps
import logging
from scipy import ndimage

# Import custom modules
try:
    from models import db, User, Prediction, ModelMetrics, init_db, create_default_users
    from preprocessing import FetalHealthPreprocessor
except ImportError:
    from src.models import db, User, Prediction, ModelMetrics, init_db, create_default_users
    from src.preprocessing import FetalHealthPreprocessor

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'static')

# Configure app
app = Flask(__name__, 
            template_folder=TEMPLATE_DIR,
            static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'fetal_health_secret_key_2026'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///../instance/fetal_health.db'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '../uploads'

ALLOWED_CSV_EXTENSIONS = {'csv'}
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('../models', exist_ok=True)
os.makedirs(os.path.join(app.static_folder, '../uploads'), exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))

# Initialize database and create default users
with app.app_context():
    try:
        init_db(app)
        create_default_users(app)
        logger.info("✅ Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")

# Paths Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')

def get_model_path(model_name):
    """Get absolute path to model file"""
    return os.path.join(MODELS_DIR, model_name)



# Load models
def load_models():
    """Load trained models"""
    try:
        models = {}
        
        ann_path = get_model_path('ann_model.h5')
        cnn_path = get_model_path('cnn_model.h5')
        scaler_path = get_model_path('scaler.pkl')
        
        if os.path.exists(ann_path):
            models['ann'] = tf.keras.models.load_model(ann_path)
            logger.info("ANN model loaded successfully")
        else:
            logger.warning(f"ANN model not found at {ann_path}")
            
        if os.path.exists(cnn_path):
            models['cnn'] = tf.keras.models.load_model(cnn_path)
            logger.info("CNN model loaded successfully")
        else:
            logger.warning(f"CNN model not found at {cnn_path}")
            
        if os.path.exists(scaler_path):
            models['scaler'] = joblib.load(scaler_path)
            logger.info("Scaler loaded successfully")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")
            
        if models:
            logger.info(f"Models loaded: {list(models.keys())}")
        else:
            logger.error("No models loaded!")
            
        return models
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return {}

models_dict = load_models()

# Decorators
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            flash('❌ Admin access required', 'error')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/user/edit/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def edit_user(user_id):

    user = User.query.get_or_404(user_id)

    user.username = request.form.get('username')
    user.role = request.form.get('role')

    password = request.form.get('password')

    if password:
        user.set_password(password)

    db.session.commit()

    flash('User updated successfully', 'success')

    return redirect(url_for('admin_users'))

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):

    user = User.query.get_or_404(user_id)

    if user.id == current_user.id:

        flash('You cannot delete yourself', 'danger')

        return redirect(url_for('admin_users'))

    db.session.delete(user)

    db.session.commit()

    flash('User deleted successfully', 'success')

    return redirect(url_for('admin_users'))

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def ensure_numeric_dataframe(df):
    """Convert dataframe columns to numeric, handle NaN values, and validate."""
    numeric_df = df.copy()
    
    for col in numeric_df.columns:
        # Convert to numeric, coercing errors to NaN
        converted = pd.to_numeric(numeric_df[col], errors='coerce')
        numeric_df[col] = converted
    
    # Fill any NaN values (missing data) with the column mean
    # This is a common strategy for handling missing numeric values
    for col in numeric_df.columns:
        if numeric_df[col].isna().any():
            # Fill with column mean, or 0 if all values are NaN
            fill_value = numeric_df[col].mean()
            if pd.isna(fill_value):
                fill_value = 0
            numeric_df[col].fillna(fill_value, inplace=True)
            logger.info(f"Filled NaN values in column '{col}' with {fill_value:.2f}")
    
    return numeric_df


# Routes
@app.route('/')
def home():
    return render_template('home.html')

    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '')
        password = request.form.get('password', '')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password) and user.is_active:
            login_user(user)
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('dashboard'))
        else:
            flash('❌ Invalid username or password', 'error')
            logger.warning(f"Failed login attempt for user: {username}")
    
    # For GET requests, just show the login page
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():

    logout_user()

    session.clear()

    flash('Logged out successfully', 'success')

    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard"""
    # Get statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.created_at.desc()
    ).limit(5).all()
    
    return render_template('dashboard.html',
                         total_predictions=total_predictions,
                         recent_predictions=recent_predictions)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    """Prediction page"""
    if request.method == 'POST':
        return handle_prediction()
    
    return render_template('predict.html')

# ================================
# FIXED CSV PREDICTION CODE
# ================================

@app.route('/api/predict_csv', methods=['POST'])
@login_required
def predict_csv_api():

    try:

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # ---------------- READ CSV ----------------
        df = pd.read_csv(file)

        if df.empty:
            return jsonify({'error': 'CSV is empty'}), 400

        # Remove target column if exists
        if 'fetal_health' in df.columns:
            df = df.drop('fetal_health', axis=1)

        # ---------------- EXPECTED FEATURES ----------------
        expected_features = [
            'baseline',
            'accelerations',
            'fetal_movement',
            'uterine_contractions',
            'light_decelerations',
            'severe_decelerations',
            'prolonged_decelerations',
            'abnormal_short_term_variability',
            'mean_value_of_short_term_variability',
            'percentage_of_time_with_abnormal_long_term_variability',
            'mean_value_of_long_term_variability',
            'histogram_width',
            'histogram_min',
            'histogram_max',
            'histogram_number_of_peaks',
            'histogram_number_of_zeroes',
            'histogram_mode',
            'histogram_mean',
            'histogram_median',
            'histogram_variance',
            'histogram_tendency'
        ]

        # ---------------- COLUMN CHECK ----------------
        matching_cols = [c for c in df.columns if c in expected_features]

        if len(matching_cols) == 21:
            df = df[matching_cols]

        else:

            numeric_df = df.select_dtypes(include=['number'])

            if numeric_df.shape[1] < 21:
                return jsonify({
                    'error': f'Need 21 numeric columns. Found {numeric_df.shape[1]}'
                }), 400

            df = numeric_df.iloc[:, :21]

        # ---------------- CLEAN DATA ----------------
        df = ensure_numeric_dataframe(df)

        # ---------------- SCALE ----------------
        features = df.values

        features_scaled = models_dict['scaler'].transform(features)

        # ---------------- PREDICT ----------------
        pred_probs = models_dict['ann'].predict(
            features_scaled,
            verbose=0
        )

        class_names = [
            'Normal',
            'Suspect',
            'Pathological'
        ]

        row_predictions = []

        row_counts = {
            'Normal': 0,
            'Suspect': 0,
            'Pathological': 0
        }

        total_confidence = 0

        # ================================
        # REAL PREDICTIONS
        # ================================
        for idx, probs in enumerate(pred_probs):

            class_idx = int(np.argmax(probs))

            confidence = float(np.max(probs))

            result = class_names[class_idx]

            total_confidence += confidence

            row_counts[result] += 1

            row_predictions.append({

                'row': idx + 1,

                'prediction': result,

                'confidence': round(confidence * 100, 2),

                'probabilities': {

                    'Normal': round(float(probs[0]) * 100, 2),

                    'Suspect': round(float(probs[1]) * 100, 2),

                    'Pathological': round(float(probs[2]) * 100, 2)
                }
            })

        # ================================
        # FINAL SUMMARY
        # ================================
        total_rows = len(row_predictions)

        most_common = max(
            row_counts,
            key=row_counts.get
        )

        average_confidence = round(
            (total_confidence / total_rows) * 100,
            2
        )

        class_percentages = {

            'Normal': round(
                (row_counts['Normal'] / total_rows) * 100,
                2
            ),

            'Suspect': round(
                (row_counts['Suspect'] / total_rows) * 100,
                2
            ),

            'Pathological': round(
                (row_counts['Pathological'] / total_rows) * 100,
                2
            )
        }

        # ================================
        # SAVE DATABASE
        # ================================
        prediction = Prediction(

            user_id=current_user.id,

            prediction_type='csv',

            result=most_common,

            confidence=average_confidence / 100,

            probabilities=json.dumps(class_percentages),

            file_name=secure_filename(file.filename)
        )

        db.session.add(prediction)

        db.session.commit()

        # ================================
        # RESPONSE
        # ================================
        return jsonify({

            'prediction': most_common,

            'result': most_common,

            'confidence': average_confidence,

            'probabilities': class_percentages,

            'row_predictions': row_predictions,

            'row_counts': row_counts,

            'total_rows': total_rows
        })

    except Exception as e:

        logger.error(f"CSV Prediction Error: {str(e)}")

        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/predict_image', methods=['POST'])
@login_required
def predict_image_api():
    """API endpoint for image prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file'}), 400
        
        if 'cnn' not in models_dict:
            return jsonify({'error': 'CNN model not loaded'}), 500
        
        # Load and process image
        # Note: PIL resize takes (width, height), so (96, 128) creates 96x128 which becomes (128, 96, 3) in numpy
        img = Image.open(file).convert('RGB').resize((96, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred_probs = models_dict['cnn'].predict(img_array, verbose=0)[0]
        class_names = ['Normal', 'Suspect', 'Pathological']
        class_idx = int(np.argmax(pred_probs))
        confidence = float(np.max(pred_probs))
        result = class_names[class_idx]
        class_percentages = {       
            'Normal': round(float(pred_probs[0]) * 100, 2),
            'Suspect': round(float(pred_probs[1]) * 100, 2),
            'Pathological': round(float(pred_probs[2]) * 100, 2)
        }
        # Save prediction to database   
        prediction = Prediction(            
            user_id=current_user.id,
            prediction_type='image',
            result=result,
            confidence=confidence,
            probabilities=json.dumps(class_percentages),
            file_name=secure_filename(file.filename)
        )
        db.session.add(prediction)
        db.session.commit() 
        return jsonify({
            'prediction': result,
            'confidence': round(confidence * 100, 2),
            'probabilities': class_percentages
        })

    except Exception as e:
        logger.error(f"Image prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/csv_to_image', methods=['GET', 'POST'])
@login_required
def csv_to_image():
    """CSV to image conversion"""
    image_path = None
    message = None
    
    if request.method == 'POST':
        try:
            file = request.files.get('file')
            if not file or file.filename == '':
                message = '❌ No file selected'
            elif not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
                message = '❌ Please upload a CSV file'
            else:
                df = pd.read_csv(file)
                
                # Create visualization
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # Plot 1: Line plot
                row = df.iloc[0]
                axes[0, 0].plot(row.values, marker='o', linewidth=2, markersize=4)
                axes[0, 0].set_title('CTG Data - Line Plot', fontweight='bold')
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Bar chart
                axes[0, 1].bar(range(min(10, len(row))), row.values[:10])
                axes[0, 1].set_title('CTG Data - First 10 Features', fontweight='bold')
                
                # Plot 3: Distribution
                axes[1, 0].hist(df.iloc[:, 0].values, bins=30, edgecolor='black')
                axes[1, 0].set_title(f'Distribution - {df.columns[0]}', fontweight='bold')
                
                # Plot 4: Heatmap of correlations
                corr_matrix = df.corr().iloc[:5, :5]
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1])
                axes[1, 1].set_title('Correlation Matrix (First 5 Features)', fontweight='bold')
                
                plt.tight_layout()
                
                # Save image
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'ctg_visualization_{timestamp}.png'
                filepath = os.path.join('static', 'uploads', filename)
                plt.savefig(filepath, dpi=100, bbox_inches='tight')
                plt.close()
                
                image_path = f'/static/uploads/{filename}'
                message = f'✅ Successfully converted CSV ({len(df)} rows, {len(df.columns)} columns)'
                
                logger.info(f"CSV to image conversion successful")
        
        except Exception as e:
            message = f'❌ Error: {str(e)}'
            logger.error(f"CSV to image error: {str(e)}")
    
    return render_template('csv_to_image.html', image_path=image_path, message=message)

@app.route('/history')
@login_required
def history():
    """View prediction history"""
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(
        Prediction.created_at.desc()
    ).all()
    return render_template('history.html', predictions=predictions)

@app.route('/metrics')
@login_required
def metrics():
    """View model metrics"""
    if 'ann' not in models_dict:
        flash('❌ Models not trained yet', 'error')
        return redirect(url_for('dashboard'))
    
    # Try to load metrics
    metrics_data = {}
    metrics_path = get_model_path('metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics_data = json.load(f)
    
    # Check for confusion matrix image
    confusion_matrix_path = get_model_path('confusion_matrix.png')
    confusion_matrix_exists = os.path.exists(confusion_matrix_path)
    
    return render_template('metrics.html',
                         metrics=metrics_data,
                         confusion_matrix_exists=confusion_matrix_exists)

@app.route('/admin/users')
@login_required
@admin_required
def admin_users():
    """Admin panel - manage users"""
    users = User.query.all()
    return render_template('admin_users.html', users=users)

@app.route('/api/confusion_matrix')
@login_required
def get_confusion_matrix():
    """Get confusion matrix image"""
    confusion_matrix_path = get_model_path('confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        return send_file(confusion_matrix_path, mimetype='image/png')
    return '', 404

# ################################################################################
# ACCURACY TRACKING FEATURES
# ################################################################################

@app.route('/api/prediction/<int:prediction_id>/verify', methods=['POST'])
@login_required
def verify_prediction(prediction_id):
    """Submit ground truth label and verify prediction accuracy"""
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Check authorization - only owner or admin can verify
        if prediction.user_id != current_user.id and current_user.role != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        data = request.get_json()
        actual_label = data.get('actual_label', '').strip()
        
        if not actual_label or actual_label not in ['Normal', 'Suspect', 'Pathological']:
            return jsonify({'error': 'Invalid actual_label. Must be: Normal, Suspect, or Pathological'}), 400
        
        # Set the actual label and calculate correctness
        prediction.set_actual_label(actual_label)
        prediction.verified_by_user = current_user.id
        
        db.session.commit()
        
        accuracy_status = "✅ Correct" if prediction.is_correct else "❌ Incorrect"
        logger.info(f"Prediction {prediction_id} verified by {current_user.username}: {accuracy_status}")
        
        return jsonify({
            'success': True,
            'prediction_id': prediction_id,
            'predicted': prediction.result,
            'actual': prediction.actual_label,
            'is_correct': prediction.is_correct,
            'accuracy_status': accuracy_status,
            'confidence': prediction.confidence
        })
    
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/accuracy/summary')
@login_required
def get_accuracy_summary():
    """Get accuracy statistics for current user or all users (if admin)"""
    try:
        if current_user.role == 'admin':
            # Admin can see all predictions
            all_predictions = Prediction.query.filter_by(accuracy_verified=True).all()
        else:
            # Users see only their own
            all_predictions = Prediction.query.filter_by(
                user_id=current_user.id,
                accuracy_verified=True
            ).all()
        
        if not all_predictions:
            return jsonify({
                'total_verified': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0,
                'message': 'No verified predictions yet'
            })
        
        verified_by_type = {}
        for pred_type in ['image', 'csv']:
            type_predictions = [p for p in all_predictions if p.prediction_type == pred_type]
            if type_predictions:
                correct = sum(1 for p in type_predictions if p.is_correct)
                verified_by_type[pred_type] = {
                    'total': len(type_predictions),
                    'correct': correct,
                    'accuracy': round((correct / len(type_predictions)) * 100, 2)
                }
        
        total = len(all_predictions)
        correct = sum(1 for p in all_predictions if p.is_correct)
        overall_accuracy = round((correct / total) * 100, 2) if total > 0 else 0
        
        return jsonify({
            'total_verified': total,
            'correct': correct,
            'incorrect': total - correct,
            'accuracy': overall_accuracy,
            'by_type': verified_by_type,
            'message': f'Overall Accuracy: {overall_accuracy}% ({correct}/{total})'
        })
    
    except Exception as e:
        logger.error(f"Accuracy summary error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/unverified')
@login_required
def get_unverified_predictions():
    """Get list of predictions that need ground truth verification"""
    try:
        if current_user.role == 'admin':
            unverified = Prediction.query.filter_by(accuracy_verified=False).order_by(
                Prediction.created_at.desc()
            ).all()
        else:
            unverified = Prediction.query.filter_by(
                user_id=current_user.id,
                accuracy_verified=False
            ).order_by(Prediction.created_at.desc()).all()
        
        predictions_data = []
        for pred in unverified:
            predictions_data.append({
                'id': pred.id,
                'prediction_type': pred.prediction_type,
                'predicted_result': pred.result,
                'confidence': pred.confidence,
                'file_name': pred.file_name,
                'created_at': pred.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'username': pred.user.username if pred.user else 'Unknown'
            })
        
        return jsonify({
            'total_unverified': len(predictions_data),
            'predictions': predictions_data
        })
    
    except Exception as e:
        logger.error(f"Unverified predictions error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/accuracy_dashboard')
@login_required
def accuracy_dashboard():
    """Accuracy tracking and verification dashboard"""
    try:
        if current_user.role == 'admin':
            total_predictions = Prediction.query.count()
            verified_predictions = Prediction.query.filter_by(accuracy_verified=True).all()
            unverified = Prediction.query.filter_by(accuracy_verified=False).order_by(
                Prediction.created_at.desc()
            ).limit(10).all()
        else:
            total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
            verified_predictions = Prediction.query.filter_by(
                user_id=current_user.id,
                accuracy_verified=True
            ).all()
            unverified = Prediction.query.filter_by(
                user_id=current_user.id,
                accuracy_verified=False
            ).order_by(Prediction.created_at.desc()).limit(10).all()
        
        # Calculate stats
        correct_count = sum(1 for p in verified_predictions if p.is_correct) if verified_predictions else 0
        verified_count = len(verified_predictions)
        unverified_count = total_predictions - verified_count
        overall_accuracy = round((correct_count / verified_count * 100), 2) if verified_count > 0 else 0
        
        # Image vs CSV breakdown
        image_preds = [p for p in verified_predictions if p.prediction_type == 'image']
        csv_preds = [p for p in verified_predictions if p.prediction_type == 'csv']
        
        image_accuracy = round((sum(1 for p in image_preds if p.is_correct) / len(image_preds) * 100), 2) if image_preds else 0
        csv_accuracy = round((sum(1 for p in csv_preds if p.is_correct) / len(csv_preds) * 100), 2) if csv_preds else 0
        
        stats = {
            'total_predictions': total_predictions,
            'verified_count': verified_count,
            'unverified_count': unverified_count,
            'correct_count': correct_count,
            'incorrect_count': verified_count - correct_count,
            'overall_accuracy': overall_accuracy,
            'image_accuracy': image_accuracy,
            'csv_accuracy': csv_accuracy
        }
        
        return render_template('accuracy_dashboard.html',
                             stats=stats,
                             unverified_predictions=unverified,
                             verified_count=verified_count)
    
    except Exception as e:
        logger.error(f"Accuracy dashboard error: {str(e)}")
        flash(f'❌ Error loading dashboard: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/predictions/<int:prediction_id>')
@login_required
def get_prediction_detail(prediction_id):
    """Get detailed information about a specific prediction"""
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        
        # Check authorization
        if prediction.user_id != current_user.id and current_user.role != 'admin':
            return jsonify({'error': 'Unauthorized'}), 403
        
        probs = {}
        if prediction.probabilities:
            import json
            probs = json.loads(prediction.probabilities)
        
        return jsonify({
            'id': prediction.id,
            'prediction_type': prediction.prediction_type,
            'predicted_result': prediction.result,
            'actual_label': prediction.actual_label,
            'confidence': prediction.confidence,
            'probabilities': probs,
            'file_name': prediction.file_name,
            'is_correct': prediction.is_correct,
            'accuracy_verified': prediction.accuracy_verified,
            'created_at': prediction.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'verified_at': prediction.verified_at.strftime('%Y-%m-%d %H:%M:%S') if prediction.verified_at else None,
            'notes': prediction.notes
        })
    
    except Exception as e:
        logger.error(f"Prediction detail error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ################################################################################
# DATA UPLOAD AND PREPROCESSING FEATURES
# ################################################################################

def is_graph_image(image_path):
    """Validate that image contains graph-based data (patterns, lines, charts)"""
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to grayscale for edge detection
        gray = np.mean(img_array, axis=2)
        
        # Apply simple edge detection using Sobel-like operator
        edges_vertical = np.abs(np.diff(gray, axis=0))
        edges_horizontal = np.abs(np.diff(gray, axis=1))
        
        # Pad to same size for combination
        edges_v_padded = np.pad(edges_vertical, ((0, 1), (0, 0)), mode='constant')
        edges_h_padded = np.pad(edges_horizontal, ((0, 0), (0, 1)), mode='constant')
        
        edges = edges_v_padded + edges_h_padded
        edge_density = np.sum(edges > 30) / edges.size
        
        # Graph images typically have moderate edge density
        # Too sparse: single lines without structure or blank images
        # Too dense: pure noise with no meaningful structure
        if edge_density < 0.01:  # Essentially blank
            return False, "Image does not contain graph-like structures (too sparse)"
        
        if edge_density > 0.90:  # Almost entirely noise
            return False, "Image appears to be too noisy or complex"
        
        return True, f"✅ Valid graph image (edge density: {edge_density:.2%})"
    
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_data():
    """Data upload page"""
    message = None
    upload_results = None
    
    if request.method == 'POST':
        upload_type = request.form.get('upload_type', 'data')
        
        if 'file' not in request.files:
            message = '❌ No file selected'
            return render_template('upload.html', message=message)
        
        file = request.files['file']
        if file.filename == '':
            message = '❌ No file selected'
            return render_template('upload.html', message=message)
        
        try:
            if upload_type == 'data' and allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
                # CSV data upload with preprocessing
                df = pd.read_csv(file)
                
                # Data validation
                if df.empty:
                    message = '❌ CSV file is empty'
                    return render_template('upload.html', message=message)
                
                logger.info(f"📊 Data preprocessing started - Rows: {len(df)}, Columns: {len(df.columns)}")
                
                # Initialize preprocessor
                preprocessor = FetalHealthPreprocessor(scaling_method='standard')
                
                # Handle missing values
                df_processed = preprocessor.handle_missing_values(df)
                print(f"✅ Data preprocessed successfully")
                logger.info(f"✅ Data preprocessed - Shape: {df_processed.shape}")
                
                # Generate individual outputs for each record
                record_outputs = []
                for idx, row in df_processed.iterrows():
                    record_outputs.append({
                        'record_id': idx + 1,
                        'features': row.to_dict(),
                        'processed': True
                    })
                
                # Save preprocessed data
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                processed_filename = f'processed_data_{timestamp}.csv'
                processed_filepath = os.path.join(UPLOADS_DIR, processed_filename)
                df_processed.to_csv(processed_filepath, index=False)
                
                message = f'✅ Data preprocessed successfully!\n' \
                         f'   📈 Rows: {len(df_processed)} | Columns: {len(df_processed.columns)}\n' \
                         f'   💾 Saved: {processed_filename}\n' \
                         f'   📋 Individual outputs generated for all {len(record_outputs)} records'
                
                logger.info(f"Successfully uploaded and preprocessed data: {len(df_processed)} rows, {len(df_processed.columns)} columns")
                
                upload_results = {
                    'type': 'data',
                    'filename': processed_filename,
                    'rows': len(df_processed),
                    'columns': len(df_processed.columns),
                    'record_outputs_count': len(record_outputs),
                    'sample_records': record_outputs[:5]  # Show first 5 records
                }
            
            elif upload_type == 'model' and allowed_file(file.filename, {'h5', 'pkl'}):
                # Model file upload
                filename = secure_filename(file.filename)
                file_path = get_model_path(filename)
                file.save(file_path)
                
                message = f'✅ Model file uploaded successfully!\n' \
                         f'   📁 File: {filename}\n' \
                         f'   🤖 Re-initializing models...'
                
                logger.info(f"Model file uploaded: {filename}")
                
                # Reload models
                global models_dict
                models_dict = load_models()
                
                if models_dict:
                    message += f'\n   ✅ Models reloaded: {list(models_dict.keys())}'
                    logger.info(f"Models reloaded successfully: {list(models_dict.keys())}")
                else:
                    message += f'\n   ⚠️ No models available after upload'
                
                upload_results = {
                    'type': 'model',
                    'filename': filename,
                    'models_loaded': list(models_dict.keys())
                }
            
            elif upload_type == 'image' and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
                # Image file upload with graph validation
                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOADS_DIR, f'temp_{filename}')
                file.save(temp_path)
                
                # Validate graph-based image
                is_valid, validation_msg = is_graph_image(temp_path)
                
                if not is_valid:
                    os.remove(temp_path)
                    message = f'❌ Image validation failed: {validation_msg}\n' \
                             f'   Please upload graph-based data (CTG charts, signals, etc.)'
                    logger.warning(f"Image validation failed: {validation_msg}")
                else:
                    # Valid graph image - save it
                    final_path = os.path.join(UPLOADS_DIR, filename)
                    os.rename(temp_path, final_path)
                    
                    message = f'✅ Image uploaded and validated successfully!\n' \
                             f'   🖼️ File: {filename}\n' \
                             f'   📊 {validation_msg}'
                    
                    logger.info(f"Image uploaded and validated: {filename}")
                    
                    upload_results = {
                        'type': 'image',
                        'filename': filename,
                        'validation': validation_msg,
                        'path': final_path
                    }
            
            else:
                message = '❌ Invalid file type. Allowed: CSV, H5, PNG, JPG'
        
        except Exception as e:
            message = f'❌ Upload error: {str(e)}'
            logger.error(f"Upload error: {str(e)}")
    
    return render_template('upload.html', message=message, upload_results=upload_results)

@app.route('/api/upload_data', methods=['POST'])
@login_required
def upload_data_api():
    """API endpoint for CSV data upload with preprocessing"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename, ALLOWED_CSV_EXTENSIONS):
            return jsonify({'error': 'Invalid CSV file'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        if df.empty:
            return jsonify({'error': 'CSV file is empty'}), 400
        
        # Preprocess data
        preprocessor = FetalHealthPreprocessor(scaling_method='standard')
        df_processed = preprocessor.handle_missing_values(df)
        
        # Generate individual record outputs
        record_outputs = []
        for idx, row in df_processed.iterrows():
            record_outputs.append({
                'record_id': idx + 1,
                'features': row.to_dict(),
                'status': 'preprocessed'
            })
        
        # Save preprocessed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_filename = f'processed_data_{timestamp}.csv'
        processed_filepath = os.path.join(UPLOADS_DIR, processed_filename)
        df_processed.to_csv(processed_filepath, index=False)
        
        logger.info(f"User {current_user.username} - Data preprocessed: {len(df_processed)} rows")
        
        return jsonify({
            'success': True,
            'message': f'✅ Data preprocessed successfully',
            'filename': processed_filename,
            'rows': len(df_processed),
            'columns': len(df_processed.columns),
            'record_outputs': record_outputs,
            'total_records': len(record_outputs)
        })
    
    except Exception as e:
        logger.error(f"Data preprocessing error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_image', methods=['POST'])
@login_required
def upload_image_api():
    """API endpoint for image upload with graph validation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Save temporarily and validate
        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOADS_DIR, f'temp_{filename}')
        file.save(temp_path)
        
        # Validate graph-based image
        is_valid, validation_msg = is_graph_image(temp_path)
        
        if not is_valid:
            os.remove(temp_path)
            return jsonify({
                'error': validation_msg,
                'success': False
            }), 400
        
        # Valid image - move to permanent location
        final_path = os.path.join(UPLOADS_DIR, filename)
        os.rename(temp_path, final_path)
        
        logger.info(f"User {current_user.username} - Image uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'message': '✅ Image successfully validated and uploaded',
            'filename': filename,
            'validation': validation_msg,
            'path': final_path
        })
    
    except Exception as e:
        logger.error(f"Image upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload_model', methods=['POST'])
@login_required
@admin_required
def upload_model_api():
    """API endpoint for model file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename, {'h5', 'pkl'}):
            return jsonify({'error': 'Invalid model file. Allowed: H5, PKL'}), 400
        
        filename = secure_filename(file.filename)
        file_path = get_model_path(filename)
        file.save(file_path)
        
        # Reload models
        global models_dict
        models_dict = load_models()
        
        logger.info(f"User {current_user.username} - Model uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'message': '✅ Model uploaded successfully',
            'filename': filename,
            'models_loaded': list(models_dict.keys())
        })
    
    except Exception as e:
        logger.error(f"Model upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)