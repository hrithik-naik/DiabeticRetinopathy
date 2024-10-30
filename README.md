# Diabetic Retinopathy Detection

This project is a web application built using Flask that supports multiple machine learning models for detecting and assessing diabetic retinopathy. The application processes medical images of the retina to detect clots, perform image segmentation, and classify the severity of diabetic retinopathy. The backend includes three primary models that work together for comprehensive analysis.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
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
- Clot detection using a custom-trained ResNet18 model.
- Image segmentation with a VGG-based U-Net for identifying specific retinal areas.
- Severity classification with an ensemble model combining EfficientNet and ViT.
- Flask-based REST API to integrate all models.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hrithik-naik/DiabeticRetinopathy.git
   cd DiabeticRetinopathy
