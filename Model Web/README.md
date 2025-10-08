# 🔭 Exoplanet Hunter AI

An AI-powered system for automatically detecting and classifying exoplanet transit signals from NASA's Kepler, K2, and TESS missions.

## 🎯 Problem Statement

Space missions like Kepler, K2, and TESS generate massive amounts of photometric data containing subtle transit signals indicating exoplanets. Traditional manual analysis is slow and not scalable. This project addresses the challenge by creating an automated AI/ML system that can:

- **Classify transit signals** into confirmed exoplanets, planetary candidates, or false positives
- **Process light curves** from multiple NASA missions
- **Provide interactive web interface** for researchers and enthusiasts
- **Continuously improve** through human feedback

## 🚀 Solution Overview

### Key Features

- **Multi-Mission Support**: Trained on Kepler, K2, and TESS datasets
- **Advanced ML Models**: XGBoost, LightGBM, Random Forest, SVM, and more
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Interactive Web App**: Upload light curves or enter parameters manually
- **Real-time Predictions**: Instant classification with confidence scores
- **Visual Analytics**: Phase-folded plots and feature visualizations

### Model Performance

Our best performing model (XGBoost) achieves:
- **Accuracy**: 75.44%
- **F1-Score**: 75.94%
- **Precision**: 77.03%
- **Recall**: 75.27%

## 📊 Dataset Information

### Training Data
- **Kepler KOI**: 9,564 objects
- **K2 Candidates**: 4,004 objects  
- **TESS TOI**: 7,699 objects
- **Total**: 21,267 exoplanet objects
- **Training Samples**: 16,917
- **Test Samples**: 4,230

### Feature Engineering
- **Basic Features**: Period, duration, depth, planet radius
- **Derived Features**: Duty cycle, transit frequency, planet radius from depth
- **Mission Encoding**: Categorical encoding for mission type
- **Score Integration**: Kepler confidence scores when available

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- macOS (with Homebrew for OpenMP)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd NASA
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas scikit-learn xgboost lightgbm matplotlib plotly streamlit joblib
   ```

4. **Install OpenMP (macOS)**
   ```bash
   brew install libomp
   ```

5. **Train the model**
   ```bash
   python simple_train.py
   ```

6. **Launch the web app**
   ```bash
   streamlit run app/StreamlitApp.py --server.port 8504
   ```

## 🎮 Usage

### Web Interface

1. **Open your browser** to `http://localhost:8504`
2. **Upload a light curve** (CSV with 'time' and 'flux' columns, or FITS file)
3. **Or enter parameters manually** in the sidebar
4. **View predictions** with confidence scores and visualizations

### Programmatic Usage

```python
import joblib

# Load the trained model
model_data = joblib.load('models/best_model.joblib')
model = model_data['model']
feature_columns = model_data['feature_columns']
label_encoder = model_data['label_encoder']

# Make predictions
features = {
    'period_days': 5.0,
    'duration_hours': 3.0,
    'depth_ppm': 1000.0,
    # ... other features
}

prediction = model.predict([list(features.values())])
class_name = label_encoder.inverse_transform(prediction)[0]
```

## 📁 Project Structure

```
NASA/
├── app/
│   └── StreamlitApp.py          # Web application
├── src/
│   ├── data/
│   │   └── process_datasets.py  # Data processing pipeline
│   ├── models/
│   │   └── train_models.py      # Model training
│   └── features/
│       └── lightcurve.py        # Light curve processing
├── dataset/                     # Raw NASA datasets
├── models/                      # Trained models
├── data/processed/              # Processed data
├── simple_train.py              # Simplified training script
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🔬 Technical Details

### Model Architecture

We trained and compared 6 different models:

1. **XGBoost** (Best) - Gradient boosting with advanced regularization
2. **LightGBM** - Fast gradient boosting framework
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Traditional gradient boosting
5. **SVM** - Support Vector Machine with RBF kernel
6. **Logistic Regression** - Linear classifier with regularization

### Feature Engineering

- **Transit Parameters**: Period, duration, depth
- **Physical Properties**: Planet radius, stellar parameters
- **Derived Metrics**: Duty cycle, transit frequency
- **Mission Context**: Encoded mission type (Kepler/K2/TESS)

### Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🎯 Classification Classes

- **Planet**: Confirmed exoplanet with high confidence
- **Candidate**: Potential exoplanet requiring further study
- **False Positive**: Not an exoplanet (stellar variability, instrumental noise, etc.)

## 🔄 Continuous Improvement

The system is designed for continuous learning:

1. **Human Feedback**: Users can correct predictions
2. **Model Retraining**: Periodic retraining with new data
3. **Feature Enhancement**: Adding new derived features
4. **Ensemble Methods**: Combining multiple models

## 🚀 Future Enhancements

- **Deep Learning**: CNN/RNN models for raw light curve analysis
- **Multi-Planet Systems**: Detection of multiple transiting planets
- **Real-time Processing**: Live data stream analysis
- **Advanced Visualization**: 3D phase space plots, interactive dashboards
- **API Integration**: REST API for programmatic access

## 📊 Results Summary

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| XGBoost | 75.44% | 77.03% | 75.27% | **75.94%** |
| LightGBM | 75.39% | 76.88% | 75.27% | 75.90% |
| Gradient Boosting | 75.22% | 76.89% | 75.06% | 75.76% |
| Random Forest | 74.96% | 76.37% | 74.92% | 75.51% |
| SVM | 70.73% | 71.86% | 71.44% | 71.47% |
| Logistic Regression | 66.64% | 70.65% | 65.19% | 66.06% |

### Key Achievements

- ✅ **Successfully merged** 3 NASA datasets (21,267 objects)
- ✅ **Trained 6 different models** with comprehensive evaluation
- ✅ **Achieved 75.94% F1-score** with XGBoost
- ✅ **Created interactive web app** for real-time predictions
- ✅ **Implemented proper train/test splits** with stratification
- ✅ **Generated detailed evaluation metrics** and visualizations

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Code style and standards
- Testing requirements
- Documentation updates
- Feature proposals

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NASA Exoplanet Archive** for providing the datasets
- **Kepler, K2, and TESS missions** for the incredible data
- **Open source community** for the amazing ML libraries
- **Space Apps Challenge** for inspiring this project

## 📞 Contact

For questions, suggestions, or collaboration opportunities, please reach out through:
- GitHub Issues
- Email: [your-email@domain.com]
- LinkedIn: [your-linkedin-profile]

---

**🔭 Happy Exoplanet Hunting!** 🚀