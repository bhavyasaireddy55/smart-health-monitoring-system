🧠 AI Smart Health Monitoring System

📌 Overview
This project is an AI-powered health monitoring platform that analyzes wearable sensor data to detect human activities and generate health insights.
The system uses ensemble machine learning and deep learning models to improve activity recognition accuracy.

🚀 Features

✔ Real-time health monitoring dashboard
✔ Human activity recognition using wearable sensor data
✔ Ensemble AI models (RandomForest + LSTM + CNN)
✔ FastAPI backend for AI inference
✔ Interactive Streamlit dashboard
✔ Health analytics using Plotly visualizations

🏗 System Architecture
Wearable Sensor Data
        ↓
Data Preprocessing
        ↓
Feature Engineering
        ↓
AI Models
(RandomForest + LSTM + CNN)
        ↓
FastAPI Backend API
        ↓
Streamlit Dashboard
        ↓
Health Insights & Predictions


📂 Project Structure
smart_health_monitoring
│
├── backend
│   └── app.py
│
├── dataset
│   └── processed_sensor_data.csv
│
├── feature_engineering
│   └── feature_selection.py
│
├── model
│   ├── train_model.py
│   ├── train_lstm.py
│   ├── train_cnn_model.py
│   ├── activity_model.pkl
│   ├── lstm_model.h5
│   └── cnn_model.h5
│
├── frontend
│   └── dashboard.py
│
├── notebooks
│
└── README.md

🤖 Machine Learning Models

The system uses ensemble learning combining three models:
| Model        | Purpose                             |
| ------------ | ----------------------------------- |
| RandomForest | Structured feature classification   |
| LSTM         | Time-series sensor pattern learning |
| CNN          | Motion pattern detection            |
Final prediction is generated using majority voting.

📈 Dashboard Features
• Real-time health monitoring
• Heart rate analytics
• Oxygen level monitoring
• AI activity prediction
• Health insights and alerts

📚 Dataset
This project uses the UCI Human Activity Recognition (HAR) dataset, which contains accelerometer and gyroscope sensor data collected from wearable devices.
