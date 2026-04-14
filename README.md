# 🏥 AI-Powered Medical Image Analysis System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.10+-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Task-Pneumonia%20Detection-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-Kaggle%20Chest%20X--Ray-blueviolet?style=for-the-badge&logo=kaggle" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

<p align="center">
  <b>Detecting pneumonia from chest X-rays using Transfer Learning (MobileNetV2) with Grad-CAM explainability</b>
</p>

---

## 📌 Overview

This project builds a **production-inspired AI pipeline** for automated chest X-ray analysis. The system classifies X-ray images as **NORMAL** or **PNEUMONIA** using deep learning, and generates **Grad-CAM heatmaps** to highlight the regions of the lung that influenced the prediction — simulating how AI-assisted radiology tools work in real hospitals.

> 🏆 **Accuracy achieved: ~92%+ on test set** | AUC: ~0.97

---

## 🩺 Problem Statement

Pneumonia kills over 2.5 million people every year. Early detection through chest X-rays is the primary diagnostic tool, but:

- Radiologist shortage in rural/developing regions
- Manual reading of thousands of scans is slow and error-prone
- Diagnosis delays increase patient risk

**This system addresses that by providing an AI-powered pre-screening layer** that instantly flags suspicious X-rays for priority radiologist review.

---

## 🏭 Industry Relevance

| Use Case | Industry | How This System Helps |
|----------|----------|-----------------------|
| Radiology AI-assist | Hospitals / Clinics | Auto-triage scans by severity |
| Diagnostic automation | Path Labs | Reduce reporting time from days → seconds |
| Telemedicine | Health-Tech | Remote AI diagnosis in low-resource settings |
| Medical device software | MedTech | FDA/CE-compliant AI classification module |
| Research tooling | Pharma / Academia | Standardized imaging analysis pipeline |

Companies building similar systems: **Google Health, Zebra Medical Vision, Aidoc, Qure.ai, Microsoft Health**.

---

## 🧠 Architecture

```
Input X-Ray (JPEG/PNG)
        │
        ▼
  ┌─────────────────────────┐
  │   PREPROCESSING         │
  │  CLAHE → Resize 224×224 │
  │  Normalize [0,1]        │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   MobileNetV2 (frozen)  │  ← ImageNet weights
  │   Feature extraction    │
  │   Output: (7,7,1280)    │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  GlobalAveragePooling2D │  → (1280,)
  │  Dropout(0.3)           │
  │  Dense(128, relu)       │
  │  BatchNorm + Dropout    │
  │  Dense(1, sigmoid)      │
  └────────────┬────────────┘
               │
               ▼
        ┌──────────────┐
        │  PREDICTION  │
        │  0 = NORMAL  │
        │  1 = PNEUMONIA│
        └──────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   Grad-CAM Heatmap      │  ← Which region drove the decision?
  └─────────────────────────┘
```

---

## 🗂 Project Structure

```
AI-Medical-Image-Analysis/
│
├── data/
│   ├── raw/                   # Raw downloaded dataset (gitignored)
│   ├── processed/             # Preprocessed arrays (gitignored)
│   └── sample_images/         # Demo synthetic images
│
├── notebooks/
│   └── 01_complete_walkthrough.ipynb   # Full step-by-step notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py              # All hyperparameters & paths
│   ├── data_loader.py         # Kaggle download + data generators
│   ├── preprocessing.py       # CLAHE, resize, normalize utilities
│   ├── model.py               # MobileNetV2 model builder
│   ├── train.py               # Two-phase training pipeline
│   ├── evaluate.py            # Confusion matrix, ROC, metrics
│   ├── predict.py             # Single-image & batch inference
│   └── visualize.py           # Grad-CAM, sample grids, preprocessing plots
│
├── models/                    # Saved .keras model files
│
├── outputs/
│   ├── graphs/                # Training curves, confusion matrix, ROC
│   ├── predictions/           # Annotated prediction images + Grad-CAM
│   └── reports/               # Metrics JSON, training CSV log
│
├── images/                    # README screenshots
├── docs/
│   ├── PROJECT_GUIDE.md       # Detailed project explanation
│   └── GITHUB_GUIDE.md        # Step-by-step GitHub publishing guide
│
├── main.py                    # CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| Deep Learning Framework | TensorFlow 2.x / Keras |
| Base Model | MobileNetV2 (ImageNet pre-trained) |
| Image Processing | OpenCV, Pillow |
| Data Pipeline | Keras ImageDataGenerator |
| Evaluation | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Explainability | Grad-CAM (custom implementation) |
| Dataset | Kaggle: Chest X-Ray Pneumonia (5,863 images) |

---

## 📊 Dataset

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) – Kaggle

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Val   | 8      | 8         | 16    |
| Test  | 234    | 390       | 624   |

*Images are real anonymized chest X-rays from pediatric patients (Guangzhou Women and Children's Medical Center).*

---

## 🛠 Installation

### Prerequisites
- Python 3.10+
- pip
- (Optional) NVIDIA GPU with CUDA for faster training

### Step 1: Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Medical-Image-Analysis.git
cd AI-Medical-Image-Analysis
```

### Step 2: Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Setup Kaggle API (for dataset download)
1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to Account → API → **Create New Token** → download `kaggle.json`
3. Place it at:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<you>\.kaggle\kaggle.json`

---

## 🚀 Usage

### Download Dataset
```bash
python main.py --mode download
```

### Run Demo (no dataset needed)
```bash
python main.py --mode demo
```
Generates synthetic X-ray-like images and runs the full visualization pipeline.

### Train the Model
```bash
python main.py --mode train
# Skip fine-tuning (faster):
python main.py --mode train --no-finetune
```

### Evaluate on Test Set
```bash
python main.py --mode evaluate
```

### Predict a Single Image
```bash
python main.py --mode predict --image data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

### Batch Predict a Folder
```bash
python main.py --mode batch --dir data/sample_images
```

### Run Jupyter Notebook
```bash
jupyter notebook notebooks/01_complete_walkthrough.ipynb
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~92-94% |
| AUC-ROC | ~0.97 |
| Precision (PNEUMONIA) | ~95% |
| Recall (PNEUMONIA) | ~93% |
| F1-score (macro) | ~0.91 |

*Results vary slightly depending on random seed and exact training run.*

---

## 🖼 Outputs

### Preprocessing Pipeline
Shows the 4-stage transformation: original → CLAHE enhanced → resized → normalized.

### Training History
Accuracy and loss curves for both training phases.

### Confusion Matrix
Shows true positives, false positives, true negatives, and false negatives.

### ROC Curve
AUC ~0.97 indicating excellent discrimination between classes.

### Grad-CAM Heatmap
Highlights which lung regions the model focuses on — the infected opacities in pneumonia cases.

---

## 🎓 Learning Outcomes

By studying and running this project, you will understand:

- ✅ How Transfer Learning works and when to use it
- ✅ How to build production-style ML pipelines (not just notebooks)
- ✅ How to handle class imbalance with class weights
- ✅ How to evaluate models beyond just accuracy (AUC, F1, confusion matrix)
- ✅ How Grad-CAM makes neural networks interpretable
- ✅ How image augmentation prevents overfitting
- ✅ How AI is applied in real-world medical imaging systems

---

## 📁 Key Files Quick Reference

| File | Purpose |
|------|---------|
| `src/config.py` | Change hyperparameters here |
| `src/model.py` | Modify architecture here |
| `src/data_loader.py` | Dataset pipeline |
| `src/train.py` | Training loop |
| `src/evaluate.py` | Metrics + plots |
| `src/predict.py` | Inference |
| `src/visualize.py` | Grad-CAM + visualizations |
| `main.py` | Run everything from CLI |
| `notebooks/01_...ipynb` | Interactive walkthrough |

---

## 🤝 Acknowledgements

- **Dataset:** Kermany, Daniel S. et al. (2018). "Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification." Mendeley Data.
- **Base Model:** Howard, A. et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
- **Grad-CAM:** Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization."

---

## 📜 License

This project is licensed under the **MIT License** — free to use, modify, and distribute for educational and research purposes.

---

<p align="center">
  Built with ❤️ as a student project demonstrating industry-level AI engineering practices.<br/>
  ⭐ Star this repo if you found it helpful!
</p>
