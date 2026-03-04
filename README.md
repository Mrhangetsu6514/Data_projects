# 🧪 Data Science Case Study: Cat vs. Dog Image Classification

This project implements a Binary Image Classifier using Deep Learning and Transfer Learning. It serves as a study on leveraging pre-trained architectures (MobileNetV2) to achieve high-accuracy results on unstructured image data with minimal training time.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](PASTE_YOUR_COLAB_URL_HERE)

## 🎯 Project Objective
The goal was to build a model capable of distinguishing between cats and dogs with an acceptable accuracy of **>98%**. The project explores the impact of **Data Augmentation** and **Global Average Pooling** on preventing overfitting in small-to-medium datasets.

## 🔬 Methodology & Pipeline

### 1. Data Ingestion & EDA
* **Source:** Kaggle `tongpython/cat-and-dog` dataset.
* **Exploration:** Random sampling was used to visualize the dataset, ensuring proper class labeling (Cat vs. Dog) before processing.

### 2. Preprocessing & Augmentation
To improve model generalization, the following spatial transformations were applied to the training set:
* **Normalization:** Pixel scaling to `1./255`.
* **Augmentation:** Random rotations (40°), width/height shifts, shearing, and zooming.
* **Validation Setup:** Validation data remained "clean" (rescaled only) to ensure an unbiased performance check.

### 3. Model Architecture (Transfer Learning)
Instead of building from scratch, this project uses a **Transfer Learning** strategy:
* **Base Model:** `MobileNetV2` (pre-trained on ImageNet).
* **Frozen Weights:** The base expert layers were frozen to retain specialized feature extraction capabilities.
* **Head:** A custom head consisting of `GlobalAveragePooling2D`, a `Dropout(0.2)` layer for regularization, and a final `Sigmoid` neuron for binary classification.

### 4. Training & Results
* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Checkpoint:** The model successfully tracks Training vs. Validation accuracy/loss to monitor convergence.


## 📊 Final Performance Check
The model provides a prediction interface that outputs the probability of the species:
* **Dog Threshold:** `prediction > 0.5`
* **Cat Threshold:** `prediction <= 0.5`
* **Target Accuracy:** High-confidence results (>99% goal).

## 📁 Project Structure
```text
├── notebooks/          # Cats_vs_Dogs_practice.ipynb
├── src/                # cats_vs_dogs_practice.py (Main Pipeline)
├── models/             # my_cat_dog_model.keras (Trained Weights)
└── README.md           # Project Documentation
