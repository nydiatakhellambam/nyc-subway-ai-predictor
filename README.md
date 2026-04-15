# 🚇 NYC Subway Delay Prediction using AI

An end-to-end machine learning system that predicts subway delays using real-time data.

---

## 🌟 Project Highlights

- 📡 Real-time data collection from MTA feeds
- 🧠 Machine Learning models (XGBoost & Random Forest)
- 📊 Interactive dashboard using Streamlit
- ⏱️ ~22 hours of real-time subway data collected

---

## 🧠 Problem Statement

Subway delay announcements are often vague.

> “Train delayed” — but how long?

This project answers:

👉 *“Given current conditions, how much delay should we expect?”*

---

## 🏗️ System Architecture
Realtime API → Data Collection → Feature Engineering → ML Model → Dashboard


---

## 📊 Dataset

- Source: MTA Real-time feeds
- Duration: ~22 hours
- Records: Hundreds of thousands
- Features:
  - trip_id
  - stop_id
  - arrival_time
  - delay (engineered)
  - hour
  - stop_sequence
  - previous delay

---

## ⚠️ Challenges Faced

- GTFS vs real-time mismatch
- No direct mapping between the schedule and actual arrival
- Sparse real-world labelled delay data

---

## 💡 Solution Approach

We created a **hybrid dataset**:

- Simulated realistic delay patterns
- Added time-based variation (rush hours)
- Introduced delay propagation across stops

---

## 🤖 Models Used

| Model | Purpose |
|------|--------|
| XGBoost | High performance boosting |
| Random Forest | Robust baseline |

---

## 📈 Model Performance

| Metric | Value |
|------|--------|
| MAE | ~60 seconds |
| RMSE | ~69 seconds |
| R² Score | Low (due to synthetic dataset) |

---

## 📊 Dashboard

### Features:
- Real-time delay prediction
- Adjustable inputs (hour, stop, delay)
- Estimated arrival time
- Delay visualization

---

## 📷 Dashboard Preview

![Dashboard](dashboard.png)

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt

Train model
python train_model.py

Run dashboard
streamlit run app.py
