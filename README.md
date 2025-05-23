# 🩺 Diabetes Prediction Model Using Logistic Regression

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-1.3+-green.svg)](https://pandas.pydata.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

This project implements a machine learning model to predict diabetes likelihood using logistic regression. The model analyzes various health metrics to provide accurate predictions, achieving **96% accuracy** on the test dataset.

### 🔬 Key Features
- **High Accuracy**: 96% prediction accuracy
- **Comprehensive Analysis**: Uses 8 health-related features
- **Data Preprocessing**: Handles missing values and feature scaling
- **Model Evaluation**: Complete performance metrics and visualizations

## 📊 Dataset

The dataset contains **100,000 records** with the following features:

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Patient gender (Female/Male) |
| `age` | Numerical | Patient age in years |
| `hypertension` | Binary | Hypertension status (0/1) |
| `heart_disease` | Binary | Heart disease status (0/1) |
| `smoking_history` | Categorical | Smoking history (never/No Info/current) |
| `bmi` | Numerical | Body Mass Index |
| `HbA1c_level` | Numerical | Hemoglobin A1c level |
| `blood_glucose_level` | Numerical | Blood glucose level |
| `diabetes` | Binary | Target variable (0/1) |

### 📈 Dataset Statistics
- **Total Records**: 100,000
- **Missing Values**: Handled through imputation
- **Class Distribution**: Imbalanced dataset with diabetes cases being minority class

## 🛠 Installation

### Prerequisites
Make sure you have Python 3.7+ installed on your system.

### Required Libraries
```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

### Alternative Installation
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.21.0
pandas>=1.3.0
seaborn>=0.11.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
```

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction
```

### 2. Prepare Your Dataset
Place your `diabetes_prediction_dataset.csv` file in the project directory and update the file path in the code:

```python
file_path = 'path/to/your/diabetes_prediction_dataset.csv'
```

### 3. Run the Model
```python
python diabetes_prediction.py
```

### 4. Key Code Snippets

#### Data Loading and Preprocessing
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('diabetes_prediction_dataset.csv')

# Convert categorical variables
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['smoking_history'] = df['smoking_history'].map({
    'never': 0, 'No Info': 1, 'current': 2
})

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
```

#### Model Training
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split data
X = scaled_df.drop('diabetes', axis=1)
y = scaled_df['diabetes']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

## 📊 Model Performance

### 🎯 Key Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 96% |
| **Precision** | 87% |
| **Recall** | 61% |
| **AUC Score** | 96% |

### 📈 Confusion Matrix
```
[[18129   163]
 [  660  1048]]
```

### 📋 Classification Report
```
              precision    recall  f1-score   support

         0.0       0.96      0.99      0.98     18292
         1.0       0.87      0.61      0.72      1708

    accuracy                           0.96     20000
   macro avg       0.92      0.80      0.85     20000
weighted avg       0.96      0.96      0.96     20000
```

### 📊 ROC Curve
The model achieves an excellent **AUC score of 0.96**, indicating strong discriminative ability between diabetic and non-diabetic cases.

## 📁 Project Structure

```
diabetes-prediction/
│
├── 📄 diabetes_prediction.py          # Main script
├── 📄 requirements.txt                # Dependencies
├── 📄 README.md                       # Project documentation
├── 📊 diabetes_prediction_dataset.csv # Dataset (not included)
├── 📈 roc_curve.png                   # ROC curve visualization
└── 📋 results/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── model_metrics.txt
```

## 🔍 Results Analysis

### ✅ Strengths
- **High Overall Accuracy**: 96% accuracy on test data
- **Excellent AUC Score**: 0.96 indicates strong model performance
- **Good Precision**: 87% precision for diabetes prediction
- **Robust Preprocessing**: Handles missing values effectively

### ⚠️ Areas for Improvement
- **Recall Enhancement**: Current recall of 61% could be improved
- **Class Imbalance**: Dataset has imbalanced classes
- **Feature Engineering**: Additional features could enhance performance

### 💡 Recommendations
1. **Try different algorithms**: Random Forest, SVM, or ensemble methods
2. **Handle class imbalance**: Use SMOTE or class weighting
3. **Feature selection**: Implement feature importance analysis
4. **Cross-validation**: Add k-fold cross-validation for robust evaluation

## 🔧 Advanced Usage

### Custom Prediction Function
```python
def predict_diabetes(gender, age, hypertension, heart_disease, 
                    smoking_history, bmi, hba1c, glucose):
    """
    Predict diabetes likelihood for new patient data
    """
    # Prepare input data
    input_data = [[gender, age, hypertension, heart_disease, 
                   smoking_history, bmi, hba1c, glucose]]
    
    # Scale the input
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    return prediction, probability

# Example usage
result, prob = predict_diabetes(1, 45, 1, 0, 2, 28.5, 6.5, 180)
print(f"Diabetes Prediction: {'Yes' if result else 'No'}")
print(f"Probability: {prob:.2f}")
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **Star this repository if you found it helpful!** ⭐


