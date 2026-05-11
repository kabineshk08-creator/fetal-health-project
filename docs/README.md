# Fetal Health Classification System

A complete full-stack web application for fetal health classification using machine learning and deep learning models.

## 📋 Project Overview

This system provides a comprehensive solution for classifying fetal health status using:
- **Artificial Neural Networks (ANN)** for tabular CTG data
- **Convolutional Neural Networks (CNN)** for image-based analysis
- **User Authentication** with role-based access control
- **Advanced Metrics** including confusion matrix, accuracy, precision, recall, and F1-score
- **Data Visualization** including heatmaps and distribution plots

## 🎯 Features

### Backend Features
✅ Flask web framework with SQLAlchemy ORM
✅ User authentication (Doctor/Admin roles)
✅ Prediction history tracking in database
✅ RESTful APIs for predictions
✅ Model metrics and performance tracking
✅ Comprehensive error handling and logging
✅ Secure file upload handling

### Machine Learning Features
✅ ANN model with 256-128-64-32 architecture
✅ CNN model with 3 convolutional layers
✅ Data preprocessing (missing values, scaling, feature selection)
✅ Dimensionality reduction with PCA
✅ Feature correlation analysis with heatmaps
✅ Three-class classification: Normal, Suspect, Pathological

### Frontend Features
✅ Bootstrap 5 responsive design
✅ Intuitive login/dashboard pages
✅ Real-time file upload with drag-and-drop
✅ Interactive prediction results with probability charts
✅ Prediction history viewer
✅ Model metrics dashboard
✅ CSV to image conversion tool
✅ Admin panel for user management

## 🏗️ Project Structure

```
fetal_health_project/
├── run.py                       # Main entry point
├── requirements.txt             # Python dependencies
├── src/                         # Source code
│   ├── __init__.py
│   ├── app_advanced.py          # Main Flask application
│   ├── config.py                # Configuration settings
│   ├── model.py                 # ML models (CNN, RNN, Hybrid)
│   ├── models.py                # Database models & authentication
│   ├── preprocessing.py         # Data preprocessing module
│   ├── evaluation.py            # Model evaluation with SHAP/LIME
│   └── deploy.py                # Real-time deployment module
├── tests/                       # Test files
│   ├── test_advanced.py
│   ├── test_app.py
│   ├── test_csv_prediction.py
│   └── test_summary.py
├── scripts/                     # Utility scripts
│   ├── create_sample_data.py
│   ├── generate_report.py
│   ├── train_advanced.py
│   └── setup.py
├── docs/                        # Documentation
│   └── README.md
├── data/                        # Sample data
├── models/                      # Trained model files
├── static/                      # CSS, JS, images
├── templates/                   # HTML templates
├── uploads/                     # User uploaded files
└── instance/                    # Database files
```
├── requirements.txt             # Python dependencies
│
├── models/                      # Trained models & artifacts
│   ├── ann_model.h5            # ANN model
│   ├── cnn_model.h5            # CNN model
│   ├── scaler.pkl              # Feature scaler
│   ├── metrics.json            # Model metrics
│   └── confusion_matrix.png     # Confusion matrix visualization
│
├── data/                        # Data files
│   ├── sample_ctg.csv          # Sample CTG data
│   └── images/                 # Generated images for CNN
│
├── templates/                   # HTML templates
│   ├── base.html               # Base template with navbar
│   ├── login.html              # Login page
│   ├── dashboard.html          # Main dashboard
│   ├── predict.html            # Prediction interface
│   ├── csv_to_image.html       # CSV visualization
│   ├── history.html            # Prediction history
│   ├── metrics.html            # Model metrics page
│   ├── admin_users.html        # Admin panel
│   ├── 404.html               # Error pages
│   └── 500.html
│
├── static/                      # Static files
│   ├── css/                    # Stylesheets
│   ├── js/                     # JavaScript
│   └── uploads/                # Generated visualizations
│
├── uploads/                     # User file uploads
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- 2GB RAM minimum

### Installation

1. **Clone/Extract the project**
```bash
cd fetal_health_project
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train models** (if not already trained)
```bash
python scripts/train_advanced.py
```

5. **Run the application**
```bash
python run.py
```

This will:
- Create synthetic CTG data (if needed)
- Train ANN model on tabular data
- Generate images and train CNN model
- Calculate comprehensive metrics
- Save all models and artifacts
- Start the Flask web server on http://127.0.0.1:5000

5. **Run the application**
```bash
python app_advanced.py
```

6. **Access the application**
Open browser and go to: `http://localhost:5000`

### Default Credentials

| Role   | Username | Password  |
|--------|----------|-----------|
| Admin  | admin    | admin123  |
| Doctor | doctor   | doctor123 |

## 📊 Usage Guide

### 1. Login
- Enter credentials and click Login
- Access dashboard with predictions, statistics, and quick actions

### 2. Make Predictions
**CSV Prediction:**
- Go to "Make Prediction" → CSV Prediction tab
- Upload CTG CSV file
- Click "Analyze CSV"
- View results with probability distribution

**Image Prediction:**
- Go to "Make Prediction" → Image Prediction tab
- Upload medical image
- Click "Analyze Image"
- View classified result and confidence

### 3. CSV to Image Conversion
- Go to "CSV to Image"
- Upload CSV file
- System generates 4-panel visualization:
  - Line plot
  - Bar chart
  - Distribution histogram
  - Correlation matrix heatmap
- Download the generated visualization

### 4. View Metrics
- Go to "Model Metrics"
- View ANN and CNN model performance
- See confusion matrix visualization
- Check accuracy, precision, recall, and F1-score

### 5. View History
- Go to "Prediction History"
- See all previous predictions
- Filter by type (CSV/Image)
- View detailed probability distributions

### 6. Admin Panel (Admin only)
- Go to "Admin Panel"
- Manage users
- View user roles and status
- Edit user information

## 🧠 Model Architecture

### ANN Model
```
Input Layer (21 features)
↓
Dense(256, relu) → Dropout(0.3)
↓
Dense(128, relu) → Dropout(0.3)
↓
Dense(64, relu) → Dropout(0.2)
↓
Dense(32, relu) → Dropout(0.2)
↓
Dense(3, softmax) → Output [Normal, Suspect, Pathological]
```

### CNN Model
```
Input Layer (128x96x3 images)
↓
Conv2D(32, 3×3, relu) → MaxPool(2×2) → Dropout(0.2)
↓
Conv2D(64, 3×3, relu) → MaxPool(2×2) → Dropout(0.2)
↓
Conv2D(128, 3×3, relu) → MaxPool(2×2) → Dropout(0.2)
↓
Flatten()
↓
Dense(128, relu) → Dropout(0.5)
↓
Dense(64, relu) → Dropout(0.3)
↓
Dense(3, softmax) → Output [Normal, Suspect, Pathological]
```

## 📈 Performance Metrics

The system tracks and displays:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positive rate among positive predictions
- **Recall**: Ability to identify actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🛠️ Data Preprocessing

The preprocessing pipeline includes:
1. **Missing Value Handling**: Mean imputation
2. **Feature Scaling**: StandardScaler normalization
3. **Feature Selection**: SelectKBest using f-classif
4. **Dimensionality Reduction**: Optional PCA
5. **Visualization**: Correlation heatmaps and distributions

## 🔐 Security Features

- Password hashing with Werkzeug
- Login-required decorators for all pages
- SQL injection prevention with SQLAlchemy ORM
- Secure file upload with whitelist validation
- CSRF protection ready
- Logged user activity tracking

## 📝 API Endpoints

### Authentication
- `POST /login` - User login
- `GET /logout` - User logout

### Predictions
- `POST /api/predict_csv` - CSV data prediction
- `POST /api/predict_image` - Image prediction

### Data Processing
- `POST /csv_to_image` - Convert CSV to visualization

### Views
- `GET /dashboard` - Main dashboard
- `GET /predict` - Prediction interface
- `GET /history` - Prediction history
- `GET /metrics` - Model metrics
- `GET /admin/users` - Admin panel

## 🐛 Troubleshooting

### Models not loading
```bash
# Retrain models
python train_advanced.py
```

### Database errors
```bash
# Delete the database and reinitialize
rm fetal_health.db
python app_advanced.py
```

### Dependencies issues
```bash
# Reinstall requirements
pip install --upgrade -r requirements.txt
```

## 📚 Sample Data

The project includes a sample CTG dataset with 300 records and 21 features including:
- Baseline heart rate
- Accelerations
- Decelerations (light, severe, prolonged)
- Fetal movement
- Uterine contractions
- Variability measures
- Histogram statistics

## 🎓 Training Your Own Models

To train on custom data:

1. Prepare CSV with target column 'fetal_health' (1-3)
2. Update path in `train_advanced.py`
3. Run `python train_advanced.py`

```python
trainer = FetalHealthModelTrainer(data_path='your_data.csv')
trainer.train_all_models()
```

## 📦 Dependencies

- Flask 2.3.3
- TensorFlow 2.13.0
- Pandas 2.0.3
- NumPy 1.24.3
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- Seaborn 0.12.2
- SQLAlchemy 3.0.5
- Flask-Login 0.6.2

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- Additional model architectures
- Real-world CTG dataset integration
- Ensemble model improvements
- Mobile-friendly UI
- Advanced analytics dashboard

## 📄 License

This project is open-source and available under the MIT License.

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review application logs in console
3. Check database for error logs
4. Inspect browser console for frontend errors

## 🎉 Features Roadmap

- [ ] API key authentication
- [ ] Multi-language support
- [ ] Advanced ensemble models
- [ ] Real-time model performance tracking
- [ ] Export results as PDF
- [ ] Integration with medical imaging systems
- [ ] Mobile application
- [ ] Docker containerization

---

**Version**: 1.0.0  
**Last Updated**: April 2026  
**Developed for**: Fetal Health Classification System
