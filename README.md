# Catsdogsclassifier
Developed an end-to-end deep learning image classification system using TensorFlow and MobileNetV2, achieving 98.24% test accuracy. Built an interactive Streamlit application for image upload and real-time cat vs dog prediction with confidence visualization.
# 🐱🐶 Cats vs Dogs Classifier using MobileNetV2

## Overview

An end-to-end Deep Learning image classification application built using TensorFlow, MobileNetV2 and Streamlit.

The system classifies images as Cat or Dog and provides Explainable AI visualizations using Grad-CAM. Users can upload images or capture images using a webcam for real-time prediction.

---

## Features

* Cat vs Dog Classification
* MobileNetV2 Transfer Learning
* Streamlit Web Application
* Webcam Capture Support
* Confidence Gauge Visualization
* Grad-CAM Explainable AI
* Model Comparison Dashboard
* Interactive User Interface

---

## Technologies Used

* Python
* TensorFlow
* Keras
* MobileNetV2
* Streamlit
* OpenCV
* Plotly
* NumPy
* Pandas
* Git & GitHub

---

## Model Performance

| Metric        | Value       |
| ------------- | ----------- |
| Test Accuracy | 98.24%      |
| Architecture  | MobileNetV2 |
| Input Size    | 224 x 224   |
| Classes       | 2           |

---

## Application Screenshots

### Home Page

![Home](screenshots/home.png)

### Prediction Result

![Prediction](screenshots/prediction.png)

### Grad-CAM Explainability

![GradCAM](screenshots/gradcam.png)

### Model Comparison Dashboard

![Dashboard](screenshots/dashboard.png)

---

## System Architecture

Input Image

↓

Image Preprocessing

↓

MobileNetV2

↓

Prediction Layer

↓

Confidence Score

↓

Grad-CAM Visualization

↓

Streamlit Dashboard

---

## Installation

```bash
git clone https://github.com/Yashasya-2509/Catsdogsclassifier.git

cd Catsdogsclassifier

python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

## Run Application

```bash
streamlit run app.py
```

## Future Enhancements

* Breed Classification
* Live Video Detection
* Prediction History Dashboard
* PDF Report Generation
* Cloud Deployment

## Project Description

Developed an end-to-end Deep Learning image classification system using TensorFlow and MobileNetV2, achieving 98.24% test accuracy. Built an interactive Streamlit application featuring webcam support, confidence visualization, Grad-CAM explainability, and model comparison analytics.
