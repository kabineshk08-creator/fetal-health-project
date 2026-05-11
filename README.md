# Fetal Health Prediction System

A comprehensive Machine Learning and Deep Learning project for predicting fetal health conditions using Cardiotocography (CTG) data and medical imaging analysis.

## 🎯 Project Overview

This project implements multiple AI models to classify fetal health status into three categories:
- **Normal** - Healthy fetal state
- **Suspect** - Intermediate state requiring monitoring
- **Pathological** - Abnormal state requiring immediate intervention

The system uses both **numerical CTG features** and **image-based analysis** for robust predictions.

## ✨ Features

- **Multiple Model Architectures**
  - Artificial Neural Network (ANN) for numerical features
  - Convolutional Neural Network (CNN) for image data
  - Advanced CTG-specific CNN models

- **Dual Input Processing**
  - Numerical feature extraction from CTG signals
  - Image-based CTG visualization and analysis
  - Comprehensive feature engineering (42+ features)

- **Web Interface**
  - Flask-based REST API
  - Interactive prediction dashboard
  - CSV batch prediction capability
  - Real-time prediction metrics

- **Data Processing**
  - CSV to image conversion pipeline
  - Automatic image classification
  - Data augmentation support

## 📦 Installation

### Requirements
- Python 3.8+
- TensorFlow/Keras
- NumPy, Pandas, Scikit-learn
- Flask, Matplotlib, OpenCV

### Setup

```bash
# Clone repository
git clone https://github.com/kabineshk08-creator/fetal-health-project.git
cd Fetal_health_Project1.0

# Install dependencies
pip install -r requirements.txt

# Initialize models (optional - downloads pre-trained models)
python initialize_models.py
```

## 🚀 Quick Start

### Run Web Application

```bash
python run.py
```
Access the application at `http://localhost:5000`

### Make Predictions via API

```python
# Numerical prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json

# Image-based prediction
curl -X POST http://localhost:5000/api/predict_image \
  -F "image=@ctg_image.png"
```

### Batch Prediction

```bash
python scripts/generate_report.py --input data.csv --output results.csv
```

## 📊 Project Structure

```
Fetal_health_Project1.0/
├── run.py                          # Main application entry point
├── requirements.txt                # Python dependencies
├── src/                            # Source code
│   ├── app_advanced.py            # Advanced Flask app
│   ├── models/                    # Model utilities
│   ├── preprocessing/             # Data preprocessing
│   └── utils/                     # Helper functions
├── models/                         # Pre-trained models
│   ├── ann_model.h5               # ANN model
│   ├── cnn_model.h5               # CNN model
│   └── metrics.json               # Performance metrics
├── scripts/                        # Training and utility scripts
│   ├── train_advanced.py          # Advanced training pipeline
│   ├── create_sample_data.py      # Generate test data
│   └── generate_report.py         # Batch prediction
├── data/                           # Sample data
│   ├── sample_ctg.csv             # Example CTG data
│   └── images/                    # CTG images
├── templates/                      # HTML templates
│   ├── dashboard.html             # Main dashboard
│   ├── predict.html               # Prediction interface
│   └── metrics.html               # Performance metrics
├── static/                         # CSS/JS assets
│   └── style.css
└── tests/                          # Test files
```

## 🧠 Models

### Artificial Neural Network (ANN)
- **Input:** 42 numerical CTG features
- **Architecture:** 3 hidden layers (256, 128, 64 neurons)
- **Activation:** ReLU + Softmax
- **Accuracy:** ~98%

### Convolutional Neural Network (CNN)
- **Input:** CTG images (224×224)
- **Architecture:** 4 convolutional blocks + dense layers
- **Pooling:** MaxPooling
- **Accuracy:** ~96%

### Advanced CTG CNN
- **Specialized for CTG signal analysis**
- **Optimized image preprocessing**
- **Real-time inference capability**

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| ANN | 97.8% | 97.5% | 97.2% | 97.3% |
| CNN | 95.9% | 95.8% | 95.3% | 95.5% |
| Advanced CNN | 96.5% | 96.2% | 96.1% | 96.1% |

## 🔧 Configuration

Key configuration in `run.py`:
```python
MODEL_PATH = 'models/'
UPLOAD_FOLDER = 'uploads/'
MAX_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
```

## 📝 Usage Examples

### Single Prediction

```bash
# Via web interface - upload CSV or image
# Or use the API:

python -c "
from src.models import predict
result = predict.get_prediction('Normal', [features...])
print(result)
"
```

### Batch Processing

```bash
# Convert CSV to images
python scripts/create_sample_data.py

# Train model
python scripts/train_advanced.py --model cnn --epochs 100

# Generate predictions
python scripts/generate_report.py
```

## 📚 Data Format

### CSV Input Format
```csv
baseline,accelerations,fetal_movement,uterine_contractions,...,health_status
120,0,0,0,...,Normal
130,5,3,2,...,Suspect
```

### Supported Health Labels
- `Normal` (0)
- `Suspect` (1)
- `Pathological` (2)

## 🎓 Features Used

The system analyzes 42 CTG features including:
- Baseline FHR (Fetal Heart Rate)
- Accelerations
- Fetal movements
- Uterine contractions
- Prolonged decelerations
- And 37 more clinical features

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home/Dashboard |
| `/predict` | POST | Make prediction |
| `/api/predict` | POST | JSON prediction |
| `/api/predict_image` | POST | Image-based prediction |
| `/metrics` | GET | Performance metrics |
| `/upload` | POST | Upload CSV file |
| `/history` | GET | Prediction history |

## 📦 Deployment

### Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "run.py"]
```

### Heroku

```bash
heroku create fetal-health-app
git push heroku main
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## ⚠️ Disclaimer

**This project is for educational and research purposes only.** Medical predictions should always be validated by qualified healthcare professionals. Do not rely solely on this system for clinical decisions.

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👨‍💻 Author

**kabineshk08-creator**  
Email: kabineshk08@gmail.com  
GitHub: [@kabineshk08-creator](https://github.com/kabineshk08-creator)

## 🙏 Acknowledgments

- CTG dataset sources and medical domain experts
- TensorFlow and PyTorch communities
- Open-source contributors

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact via email
- Check documentation in `/docs`

## 🔄 Version History

- **v1.0** - Initial release with ANN, CNN, and Advanced CNN models
- **v0.9** - Beta testing phase

---

**Last Updated:** May 2026  
**Repository:** https://github.com/kabineshk08-creator/fetal-health-project
