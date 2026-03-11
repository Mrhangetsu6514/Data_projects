# 🌿 PlantVillage Disease Classifier
> **A High-Performance Deep Learning System for Botanical Health Diagnosis**

[![Accuracy](https://img.shields.io/badge/Accuracy-99.40%25-brightgreen)](#)
[![Confidence](https://img.shields.io/badge/Inference_Confidence-99.79%25-blue)](#)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange)](#)

## 📊 Overview
This project identifies **38 different classes** of healthy and diseased plant leaves across multiple species. By leveraging **Transfer Learning** and **Fine-Tuning** on the MobileNetV2 architecture, the model achieves state-of-the-art diagnostic accuracy.

### 🖼️ Result Example
![Model Prediction Result]()  
*Figure 1: The model correctly identifies Tomato Yellow Leaf Curl Virus with **99.79% confidence**.*

---

## 🛠️ Technical Implementation

### 1. Data Pipeline & Optimization
* **Large-Scale Handling**: Processed over 2.7 GB of image data using a RAM-efficient pipeline.
* **Performance Tuning**: Utilized `tf.data.AUTOTUNE` and adjusted batch sizes to 16 to prevent memory crashes in Google Colab.
* **Preprocessing**: Implemented specialized scaling ($[-1, 1]$ range) to match the expectations of pre-trained MobileNetV2 weights.

### 2. Model Architecture
| Layer Type | Purpose |
| :--- | :--- |
| **MobileNetV2 (Base)** | Pre-trained feature extraction (ImageNet weights). |
| **GlobalAvgPooling** | Spatial data reduction. |
| **Dropout (0.2)** | Overfitting prevention during training. |
| **Dense (38)** | Softmax classification for 38 plant categories. |

### 3. Training Strategy
The project employed a **Two-Phase training approach**:
1. **Phase 1 (Transfer Learning)**: Trained the custom head while keeping the base model frozen.
2. **Phase 2 (Fine-Tuning)**: Unfroze the base model and re-trained with a very low learning rate ($1 \times 10^{-5}$) to achieve 99%+ accuracy.

---

## 📈 Performance Summary
* **Validation Accuracy:** 99.40%
* **Training Time:** ~20.72 minutes (Phase 1)
* **Model Size:** ~26.8 MB (Optimized `.keras` format)

---

## 🚀 How to Run
To use the diagnostic script, ensure you have the required libraries installed:

```bash
pip install tensorflow numpy matplotlib
