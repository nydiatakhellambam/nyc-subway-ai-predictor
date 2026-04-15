# 🚇 NYC Subway AI Predictor

An AI-based system that predicts subway delays using real-time data and machine learning.

---

## 📌 Features

- Real-time data collection from MTA API
- Delay prediction using XGBoost & Random Forest
- Interactive dashboard using Streamlit
- Model evaluation with MAE, RMSE, and R²

---

## 🧠 Models Used

- XGBoost Regressor
- Random Forest Regressor

---

## 📊 Performance

- MAE: ~60 seconds  
- Dataset: ~22 hours of real-time subway data  

---

## ⚠️ Challenges

- GTFS and real-time data mismatch  
- Trip ID inconsistencies  
- Required hybrid dataset approach  

---

## 💡 Solution

We created a hybrid dataset with:
- Time-based delay patterns  
- Delay propagation between stops  
- Realistic simulation of subway behavior  

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train_xgboost.py