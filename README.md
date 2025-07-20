# 🏋️‍♂️ Gym Crowdness Predictor using SGDRegressor

Welcome to the **Gym Crowdness Predictor** — a machine learning project that helps forecast gym crowd levels using time and weather data. This project is built using a **Linear Regressor** trained with **Stochastic Gradient Descent (SGD)**.

## 🚀 Motivation

I love working out, but like many gym-goers, I dislike overcrowded gyms. It's frustrating to wait for equipment, skip parts of your routine, or lose momentum during a workout.

To solve this, I combined my interest in fitness with machine learning to predict gym crowd levels and plan workouts more effectively.

## 💡 Project Objective

The goal of this project is to:

> **Predict the number of people at the gym based on the day, time, and other factors.**

This way, gym users can:
- 🕒 Plan workouts more efficiently  
- 🚫 Avoid peak hours  
- 🏃 Stick to their routines without interruptions  

## 📊 Dataset

The dataset used for this project contains:

- ✅ Over **60,000 rows**
- ✅ **11 features**, including:
  - Day of the week
  - Hour of the day
  - Temperature
  - Other contextual variables

## 🧠 Model

The core model is a **Linear Regressor** trained using **Stochastic Gradient Descent (SGDRegressor)** from `scikit-learn`. Key transformations and techniques include:

- Cyclical encoding of time-based features (hour and day)
- Polynomial feature transformation
- Data normalization using `StandardScaler`
- Model tuning using `GridSearchCV`

## 🛠️ Tech Stack

- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn (for EDA and visualization)
- Jupyter Notebook

## 📈 Results

The model offers reasonable predictive performance to identify crowded vs. less crowded times. It's not perfect, but it’s a great step toward smarter gym scheduling.
