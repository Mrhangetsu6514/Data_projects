# 🐾 Cats vs. Dogs: Computer Vision Pipeline

A deep learning project focused on Binary Image Classification. This pipeline demonstrates the process of image preprocessing, CNN architecture design, and performance validation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a1fLks9yCOr4De19AUHwIMSrj4FpdTjK?usp=sharing)

---

## 🚀 Overview
The objective is to build a robust model capable of distinguishing between images of cats and dogs using Convolutional Neural Networks (CNN).

## 🛠️ Tech Stack
* **Framework:** TensorFlow / Keras
* **Image Processing:** OpenCV / PIL
* **Environment:** Google Colab (GPU Accelerated)

## 🧠 Technical Features
* **Data Augmentation:** Used `ImageDataGenerator` to prevent overfitting.
* **CNN Architecture:** Implemented Conv2D, MaxPool2D, and Dropout layers.
* **Normalization:** Rescaled pixel values to [0-1] for better convergence.

## 📁 Dataset Structure
```text
dataset/
├── training_set/
│   ├── cats/
│   └── dogs/
└── test_set/
