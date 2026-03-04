# 🧪 Data Science Case Study: Cat vs. Dog Image Classification

This project implements a Binary Image Classifier using Deep Learning. It serves as a study on Convolutional Neural Networks (CNNs), focusing on how spatial feature extraction and data augmentation can overcome high variance in unstructured image data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a1fLks9yCOr4De19AUHwIMSrj4FpdTjK?usp=sharing)

## 🎯 The Research Goal
The objective was to develop a predictive model that identifies patterns (ears, snout shapes, fur textures) to distinguish between two species with a target accuracy of >80% using a custom-built CNN.

## 🔬 Methodology & Experimentation

### 1. Exploratory Data Analysis (EDA)
Before modeling, the data was inspected for:
* **Class Balance:** Ensuring an equal number of cat and dog images to avoid model bias.
* **Dimensions:** Analyzing various aspect ratios to determine the optimal resize target (e.g., 150x150).

### 2. Feature Engineering & Preprocessing
* **Pixel Rescaling:** All pixel values were normalized from $[0, 255]$ to $[0, 1]$ to improve the efficiency of backpropagation.
* **Data Augmentation:** To simulate a larger dataset and improve generalization, I applied random transformations:
    * Horizontal Flips
    * Rotation Range (40°)
    * Shear/Zoom ranges

### 3. Model Architecture
I designed a Sequential CNN with the following logic:
* **Convolutional Layers:** 3 layers with increasing filters (32, 64, 128) to capture increasingly complex features.
* **Activation:** `ReLU` for hidden layers to prevent vanishing gradients, and `Sigmoid` for the final output (Binary Classification).
* **Regularization:** `Dropout(0.5)` was added before the Dense layer to mitigate overfitting.



## 📊 Performance Evaluation
The model's success was measured using:
* **Binary Cross-Entropy Loss:** To measure the distance between predicted probabilities and actual labels.
* **Accuracy/Loss Curves:** Visualizing the training vs. validation delta to identify the "Sweet Spot" before overfitting occurred.

## 📂 Project Structure
```text
├── data/               # Dataset links and samples
├── notebooks/          # Colab file with step-by-step EDA and Training
├── models/             # Saved .h5 or .keras model files
└── results/            # Accuracy plots and Confusion Matrix
