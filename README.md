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
# Diabetic Retinopathy Project

## Overview

Diabetic Retinopathy (DR) is one of the leading causes of blindness worldwide, affecting nearly 1 in 3 people with diabetes. Early detection is critical, but the manual examination of retinal images is time-consuming and requires specialized expertise, especially given the subtle variations between different stages of DR. DR occurs when high blood sugar levels damage the blood vessels in the retina, leading to vision changes and potentially severe complications. In its advanced stages, DR can cause bleeding in the eye and the formation of scar tissue, which may result in permanent vision loss if not treated promptly. Additionally, other retinal conditions such as Central Retinal Vein Occlusion (CRVO), Branch Retinal Vein Occlusion (BRVO), Pre-Retinal Hemorrhage (PRH), and Retinal Hemorrhages Localized (RHL) can lead to significant vision loss if not identified early.

![Severe Diabetic Retinopathy](https://github.com/user-attachments/assets/d2f17a4d-1eb3-4411-a64f-b00d8c561b16)


This project addresses these challenges by leveraging deep learning to automatically detect and classify Diabetic Retinopathy and other retinal diseases from high-resolution images. The model classifies DR into five categories: No DR, Mild, Moderate, Severe, and Proliferative DR. It also identifies the presence of CRVO, BRVO, PRH, and RHL, providing a comprehensive assessment of a patient's retinal health.

- CRVO (Central Retinal Vein Occlusion)- occurs when the central retinal vein becomes blocked, leading to retinal hemorrhages and fluid leakage
- BRVO (Branch Retinal Vein Occlusion)-characterized by the obstruction of one of the branches of the central retinal vein, resulting in localized swelling and bleeding in the retina
- PRH (Pre-Retinal Hemorrhage)-bleeding that occurs between the retina and the inner limiting membrane
- RHL (Retinal Hemorrhages Localized)-involves the presence of localized areas of bleeding within the retina

Detecting these additional conditions is crucial, as they can exacerbate vision loss and complicate treatment plans if left undiagnosed. By improving diagnostic accuracy and efficiency, the model supports healthcare professionals in early intervention, potentially saving patients' vision and showcasing AI's transformative role in healthcare.

### Dataset
The dataset for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-2019-data), containing approximately 3,700 high-resolution retinal images categorized into five classes: No DR, Mild, Moderate, Severe, and Proliferative DR. Additionally, for analyzing clotting and bleeding in Diabetic Macular Edema (DME), we utilized a dataset available at [Zenodo](https://zenodo.org/records/7505822). This dataset provided crucial information regarding retinal hemorrhages and other relevant factors, allowing for a comprehensive understanding of the relationship between Diabetic Retinopathy and DME.

In this project, we employed a robust deep learning framework to detect and classify Diabetic Retinopathy and other retinal conditions using a custom ensemble model that combines EfficientNet and Vision Transformer (ViT) architectures. The training process began with the creation of a comprehensive dataset loader utilizing Albumentations for advanced data augmentation techniques, such as random rotations, brightness adjustments, and dropout, to enhance model robustness and improve generalization. We configured the model to classify images into five categories of Diabetic Retinopathy while also detecting other conditions like Central Retinal Vein Occlusion (CRVO), Branch Retinal Vein Occlusion (BRVO), Pre-Retinal Hemorrhage (PRH), and Retinal Hemorrhages Localized (RHL). The training utilized Focal Loss, which is particularly effective in addressing class imbalance and emphasizing harder-to-classify samples. A Cosine Annealing Warm Restarts scheduler was employed to dynamically adjust the learning rate, optimizing convergence throughout the 100 epochs of training. The model was evaluated using accuracy metrics on both training and validation datasets, with the best model weights saved for future inference. Overall, this approach not only demonstrates the effectiveness of ensemble learning in medical imaging but also highlights the potential of AI to assist in critical healthcare diagnostics related to DME and its complications.

### Ensemble learning model for DR

### Model Architecture
This project utilizes an ensemble model architecture that combines two powerful neural network architectures: *EfficientNet-B0* and the *Vision Transformer (ViT)*. EfficientNet-B0 is recognized for its efficiency regarding parameters and computational resources while achieving high accuracy in image classification tasks, thanks to its balanced scaling of depth, width, and resolution. In contrast, ViT leverages self-attention mechanisms to effectively capture global dependencies in images, making it particularly effective for various computer vision applications. By averaging the predictions from both EfficientNet and ViT, this ensemble model employs a soft voting approach that enhances robustness and accuracy, ultimately improving the classification of Diabetic Retinopathy from retinal images.

## Results
![output (1)](https://github.com/user-attachments/assets/4d58e008-7b10-4070-ba54-9dc5d3579af2)
![output](https://github.com/user-attachments/assets/cd70e467-bc33-457b-956c-1ded6809c710)
## Clot and bleed detection model
### Model Architecture

- *Dataset*: Custom ClotDataset loads images and labels, filtering only available images.
- *Data Augmentation*: Resizes to (512, 384), applies random flips, rotations, and normalization.
- *Data Split*: 80% training, 20% validation, stratified by clot labels.
- *Weighted Sampling*: Implements WeightedRandomSampler to address class imbalance.
- *Model*: Pre-trained ResNet18 with the final layer modified for binary classification.
- *Loss Function*: BCEWithLogitsLoss combines sigmoid activation and binary cross-entropy.
- *Optimizer*: Adam optimizer with a learning rate of 0.001.
- *Accuracy Calculation*: Rounds sigmoid output to compare with ground truth.

  ### Results
    
  ![clot_accuracy_graph](https://github.com/user-attachments/assets/17f6e7f5-478b-4d54-a55a-54f8b78c086f)
  
  ![clot_loss_graph](https://github.com/user-attachments/assets/7cb4f5bc-e267-463f-8bc6-9d915f47632b)










