---
title: Pest Detection
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# 🌾 Agricultural Pest Detection System

This application uses a fine-tuned MobileNetV2 model to detect agricultural pests from images.

## Features
- **Pest Identification**: Upload an image to identify the pest.
- **Model Retraining**: Upload new training data to improve the model.
- **Monitoring**: View system metrics and performance.

## How it works
This Space runs two services in a single Docker container:
1.  **FastAPI Backend**: Handles model inference and retraining logic (Port 8000).
2.  **Streamlit Frontend**: Provides the user interface (Port 7860).

## Usage
1.  Go to the **Prediction** tab.
2.  Upload an image of a pest.
3.  Click **Predict** to see the result.
