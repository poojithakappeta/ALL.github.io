# 🔬 DL-ALL: Acute Lymphoblastic Leukemia Classification

A complete end-to-end deep learning pipeline for detecting and classifying **Acute Lymphoblastic Leukemia (ALL)** from peripheral blood smear images.

---

## 📌 Highlights

- ✅ Reimplementation of the base research paper
- ✅ Novel hybrid deep learning model: **LeukVision**
- ✅ Flask-based deployed web application for real-time inference

---

## 1️⃣ Base Paper Reimplementation

**Dataset:** ALL-IDB2  
**Techniques Used:**  
- GLCM (Gray Level Co-occurrence Matrix)  
- LBP (Local Binary Patterns)  
- PCA (Principal Component Analysis)  
- DWT (Discrete Wavelet Transform)  

**Classifiers:**
- SVM  
- Random Forest  
- ResNet50  
- MobileNetV2  

### 📊 Results

| Model         | Accuracy |
|---------------|----------|
| Random Forest | 98.25%   |
| ResNet50      | 93.5%    |
| MobileNetV2   | 90.3%    |

📁 Folder: `base_paper/`

---

## 2️⃣ Novel Implementation – LeukVision

We developed a **hybrid fusion model** called **LeukVision**, combining the strengths of multiple feature extraction methods:

- 🧠 **ResNet50** (trainable CNN backbone)
- 🔍 **ViT** – Frozen Vision Transformer for global attention
- 💡 **Class-Specific Prompt Embeddings** (128-D)
- 🌊 **Wavelet Features** using Haar DWT (4096-D)

👉 All features are concatenated into a **7032-D** vector and passed through a fully connected classifier.

### 📊 Accuracy

| Model              | Accuracy |
|-------------------|----------|
| **LeukVision**     | **100%** |
| MobileNet + XGBoost | 98.25% |
| ShuffleNet + RF    | 97.75%  |
| VGG16 + SVM        | 93.75%  |

📁 Folder: `novelty_models/`

---

## 🧠 LeukVision Architecture

![LeukVision Architecture](static/leukvision_architecture.png)

📁 Image Path: Place the image in the `web_app/static/` folder.

---

## 🌐 Web Application

We built a user-friendly **Flask** web interface for real-time prediction.

### 🔍 Features:
- Upload a peripheral smear image
- **Binary Detection** – Is ALL present?
- **Multiclass Classification** – Benign, Early, Pre, Pro stages
- Option to compare predictions across all trained models

📁 Folder: `web_app/`

---

## 🛠 Run Locally

```bash
# Clone the repository
git clone https://github.com/poojithakappeta/ALL.github.io
cd ALL.github.io/web_app

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
