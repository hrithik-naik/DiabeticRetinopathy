## Diabetic Retinopathy Detection

This project is a web application built using Flask that supports multiple machine learning models for detecting and assessing diabetic retinopathy. The application processes medical images of the retina to detect clots, perform image segmentation, and classify the severity of diabetic retinopathy. The backend includes three primary models that work together for comprehensive analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [Clot Detector](#clot-detector)
  - [Segmentation Model](#segmentation-model)
  - [Ensemble Classifier](#ensemble-classifier)
- [Acknowledgements](#acknowledgements)

## Overview
Diabetic retinopathy is a complication of diabetes that affects the eyes, and early detection is crucial to prevent vision loss. This application is designed to:
1. Detect clot presence in retinal images.
2. Segment retinal areas for precise analysis.
3. Classify the severity of diabetic retinopathy.

The models are implemented in PyTorch and TensorFlow, leveraging deep learning architectures like ResNet, VGG, EfficientNet, and Vision Transformers.

## Features
- **Clot detection** using a custom-trained ResNet18 model.
- **Image segmentation** with a VGG-based U-Net for identifying specific retinal areas.
- **Severity classification** with an ensemble model combining EfficientNet and Vision Transformers (ViT).
- **Flask-based REST API** to integrate all models.
- **User-friendly interface** for easy image uploads and results display.

## Technologies Used
The technologies utilized in this project include:
- **Python**: Backend programming language.
- **Flask**: Web framework for the application.
- **TensorFlow/Keras**: Deep learning library used for model development and predictions.
- **PyTorch**: Framework for training deep learning models.
- **HTML/CSS/JavaScript**: Frontend development for user interaction.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hrithik-naik/DiabeticRetinopathy.git
   cd DiabeticRetinopathy
2. **Virtual Environment Setup**:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate
  ```
3. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```
4. **MODEL**:
Download Model Weights: Download the model weight files and place them as specified in each model class (ClotDetector, InferencePipeline, EnsembleClassifier). Note that due to the size of the weight files, they could not be included in this repository.
5.**Gemini API key**:
   in APP.py paste ur Gemini Api Key which u can create from https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://aistudio.google.com/









