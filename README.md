# 🔬 DL-ALL: Acute Lymphoblastic Leukemia Classification

A complete end-to-end deep learning pipeline for detecting and classifying Acute Lymphoblastic Leukemia (ALL) from peripheral blood smear images. This project includes:

- ✅ Reimplementation of base paper
- ✅ Novel hybrid models including **LeukVision**
- ✅ A deployed web application built using Flask

---

## 📁 Project Structure

LL.github.io/
├── base_paper/ # Base paper implementation
├── novelty_models/ # LeukVision, MobileNet+XGBoost, etc.
├── web_app/ # Deployed web code (Flask)
│ ├── app.py
│ ├── fusion_model.py
│ ├── wavelet_utils.py
│ ├── templates/
│ ├── static/
│ ├── requirements.txt
│ └── render.yaml
└── README.md

---

## 1️⃣ Base Paper Reimplementation

- Dataset: ALL-IDB2
- Techniques: GLCM, LBP, PCA, DWT
- Classifiers: SVM, Random Forest, ResNet50, MobileNetV2

### 🔹 Results:
| Model         | Accuracy |
|---------------|----------|
| Random Forest | 98.25%   |
| ResNet50      | 93.5%    |
| MobileNetV2   | 90.3%    |

📁 Folder: `base_paper/`

---

## 2️⃣ Novelty Implementation – LeukVision

We introduced a hybrid fusion model **LeukVision** that combines:

- 🧠 ResNet50 (trainable)
- 🔍 ViT (frozen Vision Transformer)
- 💡 Class-specific Prompt Embeddings (128-D)
- 🌊 Wavelet Features (4096-D Haar DWT)

These features are concatenated (7032-D) and passed through a final dense classifier.
This is the architecture of the LeukVision model ![LeukVision Architecture](static/leukvision_architecture.png)

### 🔹 Accuracy:
- **LeukVision:** 100%
- MobileNet + XGBoost: 98.25%
- ShuffleNet + RF: 97.75%
- VGG16 + SVM: 93.75%

📁 Folder: `novelty_models/`

---

## 🧠 LeukVision Architecture

> Replace with actual image file name after uploading

![LeukVision](static/leukvision_architecture.png)

---

## 🌐 Web App (Flask)

A Flask-based web interface for real-time detection and classification:

- Upload any smear image
- Detect if **ALL** is present (Binary ViT)
- Classify into: **Benign**, **Early**, **Pre**, **Pro**
- Compare across all implemented models

📁 Folder: `web_app/`

---

## 🚀 Live App

> 🟡 Deploying soon on [Render](https://render.com)  
> (Link will be added here after deployment)

---

## 🛠 Run Locally

```bash
git clone https://github.com/poojithakappeta/ALL.github.io
cd ALL.github.io/web_app
pip install -r requirements.txt
python app.py
flask
torch
torchvision
transformers
Pillow
joblib
gdown
dill
Contributors
Kappeta Poojitha

Shalini R

Bathala Vandana

SASTRA Deemed University – School of Computing
Final Year Major Project Submission (2025)

---

✅ Save this as a file named `README.md` and place it in your root repo (`ALL.github.io/`).

Let me know when you're ready to deploy or add screenshots!
