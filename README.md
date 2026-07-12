# Sales Forecasting & Demand Intelligence System

## Overview

An end-to-end Machine Learning and Time Series Forecasting project that predicts future product demand, detects unusual sales patterns, segments products based on demand behavior, and presents business insights through an interactive Streamlit dashboard.

This project simulates a real-world retail demand forecasting system used by e-commerce and retail companies to improve inventory planning, reduce stock shortages, and optimize business decisions.

## Features

### Sales Dashboard
- Total sales according to Region (West, South, East and Central)
- Total sales according to Category (Furnitur, Offlice Supplies and Technology)
- Total sales by year
- Monthly sales trend

### Forecast Explorer
- Sales forcast for next 3-months
- Forecast Results
- Forecast Trend
- Forecast according to region and category
- Model Performance(MAE, RMSE)

### Anomaly Report
- Weekly Sales with Detected Anomalies\
- Detected Anomaly Weeks
- Total anamolies
- Isolation Forest detected several unusual sales weeks that differ from the normal sales pattern

### Product Demand Segments
- Demand Segmentation (PCA Visualization)
- Product Demand Segments
- Products in Each Cluster
- groups products based on their sales volume, growth rate, volatility, and average order value.
- businesses optimize inventory, reduce storage costs, and improve stock planning.

### Other
- Interactive Streamlit Dashboard
- Sales Forecasting using:
  - SARIMA
  - Facebook Prophet
  - XGBoost
- Model Performance Comparison (MAE, RMSE, MAPE)

## Tech Stack

### Programming Language
- Python

### Data Analysis
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- XGBoost

### Time Series Forecasting
- Statsmodels (SARIMA)
- Facebook Prophet

### Clustering & Anomaly Detection
- K-Means Clustering
- PCA
- Isolation Forest
- Z-Score

### Data Visualization
- Matplotlib
- Plotly

### Web Framework
- Streamlit

### Development Tools
- Jupyter Notebook
- VS Code
- Git & GitHub

## ⚙️ How It Works

### Step 1 — Data Preparation
- Load Superstore Sales dataset
- Parse date columns
- Handle missing values
- Feature engineering
- Aggregate weekly and monthly sales

### Step 2 — Exploratory Data Analysis
- Sales trend analysis
- Revenue by category
- Regional sales comparison
- Seasonal pattern identification

### Step 3 — Time Series Analysis
- Monthly sales visualization
- Time Series Decomposition
- Stationarity testing using ADF Test
- Differencing (if required)

### Step 4 — Forecasting Models
Three different forecasting approaches were implemented:
- SARIMA
- Facebook Prophet
- XGBoost

Each model was evaluated using:
- MAE
- RMSE
- MAPE

XGBoost achieved the best performance and was selected as the production model.

### Step 5 — Product & Region Forecasting
Generated separate forecasts for:
- Furniture
- Technology
- Office Supplies
- West Region
- East Region

### Step 6 — Anomaly Detection
Detected abnormal sales patterns using:
- Isolation Forest
- Z-Score Analysis

### Step 7 — Product Demand Segmentation
Products were grouped into demand segments using:
- K-Means Clustering
- PCA Visualization

### Step 8 — Interactive Dashboard
Developed a multi-page Streamlit dashboard featuring:
- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

## Dashboard Screenshots

### Sales Overview Dashboard

<img width="1917" height="895" alt="image" src="https://github.com/user-attachments/assets/af6bb3c1-2359-4fa3-a753-131a8cb78d39" />

### Forecast Explorer

<img width="1912" height="887" alt="image" src="https://github.com/user-attachments/assets/e2b50d0b-bad4-4393-b5ef-e8c3272b8020" />

### Anomaly Report

<img width="1913" height="877" alt="image" src="https://github.com/user-attachments/assets/9ce3bf12-7e66-49b0-86ff-a613939ffc4e" />

### Product Demand Segments

<img width="1912" height="882" alt="image" src="https://github.com/user-attachments/assets/3748b8a3-73ba-42ad-9a16-a0fa26ae1586" />

## Installation

Clone the repository

```bash
git clone https://github.com/arelli-sanjay/Sales-Forecasting-Demand-Intelligence-System.git
```

Move into the project directory

```bash
cd SalesForecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

## Live Demo

*Streamlit App:**  
https://sales-forecasting-demand-intelligence-systemgit-2ngtjokutumcbi.streamlit.app/

- Explore the interactive Sales Forecasting & Demand Intelligence Dashboard with forecasting, anomaly detection, and product demand segmentation.

## What I Learned

Through this project, I gained practical experience in:
- Time Series Forecasting
- SARIMA Modeling
- Facebook Prophet
- XGBoost Regression
- Feature Engineering for Time Series
- Forecast Model Evaluation
- Time Series Decomposition
- Stationarity Testing (ADF Test)
- Isolation Forest for Anomaly Detection
- K-Means Clustering
- PCA Visualization
- Interactive Dashboard Development using Streamlit
- Model Deployment Workflow
- Organizing production-ready Machine Learning projects

## Author

**ARELLI SANJAY**
- GitHub: https://github.com/arelli-sanjay
- Linkedin: https://www.linkedin.com/in/sanjay-arelli-2b0970383/

## Support
If you like Sales Forecasting & Demand Intelligence System consider giving it a star, on GitHub.
