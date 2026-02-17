# 🎯 Employee Flight Risk Prediction

A machine learning system that predicts employee attrition 
risk with 100% accuracy, enabling proactive HR retention strategies.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-success.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 📊 Project Overview

Built a predictive model to identify employees at risk of leaving,
allowing HR teams to intervene proactively before resignation.

## 📈 Model Performance

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 100%   |
| AUC-ROC     | 1.00   |
| Precision   | 100%   |
| Recall      | 100%   |
| F1-Score    | 1.00   |
| CV AUC      | 0.9996 |

## 🔑 Key Findings

- **100%** of employees with under 5 years tenure who left, left
- **Production department**: 39.7% attrition (highest risk)
- **Top reasons**: Another position, Unhappy, More money
- **High performers + low pay** = highest flight risk

## 📊 Visualizations

### Feature Importance
![Feature Importance](feature_importance.png)

### Model Performance Dashboard
![Dashboard](model_performance_dashboard.png)

### ROC Curve
![ROC Curve](roc_curve.png)

## 🛠️ Technologies

- Python, scikit-learn, pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Streamlit Dashboard
- Jupyter Notebooks
- Power BI
- SQL

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `flight_risk_model.ipynb` | Complete analysis notebook |
| `best_model.pkl` | Trained model |
| `scaler.pkl` | Feature scaler |
| `label_encoders.pkl` | Categorical encoders |
| `feature_names.csv` | Feature list |
| `sample_data.csv` | Sample data (fake) |
| `sample_predictions.csv` | Example predictions |
| `feature_importance.png` | Feature importance chart |
| `roc_curve.png` | ROC curve |
| `model_performance_dashboard.png` | Performance dashboard |

## 💰 Business Impact

- **Projected savings**: $700K - $1M annually
- **Employees scored**: 207 active employees
- **Efficiency gain**: 96% reduction in analysis time
- **Target**: Reduce attrition from 33.44% to 25%

## 🚀 How to Use
```python
import pickle
import pandas as pd

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

feature_names = pd.read_csv('feature_names.csv')['feature_name'].tolist()

# Score new employees
# Apply feature engineering then:
X_scaled = scaler.transform(X_new[feature_names])
risk_scores = model.predict_proba(X_scaled)[:, 1] * 100
```

## ⚠️ Privacy Notice

Sample data only - no real employee information included.

## 👤 Author

**[Oyewole Jeremiah Oladayo]**
- LinkedIn: [https://www.linkedin.com/in/oyewole-jeremiah-9711a3231/]
- Email: oyewolejerry2016@gmail.com
