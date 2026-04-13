# Human Activity Recognition (HAR) using Machine Learning & Deep Learning

A complete end-to-end pipeline for recognizing human physical activities from smartphone sensor data using the UCI HAR Dataset.

---

## 📌 Project Overview

This project implements a modular Human Activity Recognition system that classifies six physical activities using accelerometer and gyroscope data collected from a waist-mounted smartphone. The pipeline covers preprocessing, dimensionality reduction, classification, clustering, and deep learning.

**Activities Recognized:**
- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## 📂 Project Structure

    HAR-Project/
    │
    ├── main.py               # Main pipeline runner
    ├── preprocessing.py      # Data cleaning, normalization, PCA
    ├── classification.py     # ML classification models
    ├── clustering.py         # Clustering algorithms
    ├── neural_network.py     # PyTorch deep learning model
    └── requirements.txt      # Dependencies
---

## 📊 Dataset

**UCI Human Activity Recognition Dataset**
- 30 subjects aged 19–48
- Samsung Galaxy S II smartphone mounted on waist
- Sensors: 3-axis accelerometer + 3-axis gyroscope at 50Hz
- 10,299 total samples (7,352 train / 2,947 test)
- 561 original features → reduced to 106 after preprocessing

Dataset source: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

---

## ⚙️ Preprocessing Pipeline

1. **Data Cleaning** — Removed 84 duplicate features and 262 highly correlated features (threshold > 95%)
2. **Normalization** — Standard scaling (zero mean, unit variance)
3. **Dimensionality Reduction** — PCA retaining 95% variance, reducing 215 → 106 features

---

## 🤖 Classification Models & Results

| Model | Accuracy |
|---|---|
| Rule-Based Classifier | 28.47% |
| Decision Tree | 71.06% |
| Naive Bayes | 80.42% |
| MLP (scikit-learn) | 92.70% |
| Proposed PyTorch Neural Network | 92.50% |
| SVM (RBF Kernel) | **94.30%** |

### Proposed PyTorch Neural Network Architecture
- Input layer (106 features)
- Hidden layer 1 (128 neurons, ReLU)
- Hidden layer 2 (64 neurons, ReLU)
- Output layer (6 classes)
- Trained with Adam optimizer, CrossEntropyLoss, 30 epochs, batch size 32

---

## 🔍 Clustering Results

| Method | Silhouette Score | Davies-Bouldin | Dunn Index |
|---|---|---|---|
| K-Means | 0.084 | 3.041 | 0.076 |
| Agglomerative (Ward) | 0.058 | 3.527 | 0.064 |
| DBSCAN | — | — | — |
| Hybrid Distance-Filtered KMeans *(proposed)* | 0.077 | 3.170 | 0.067 |

The proposed Hybrid Distance-Filtered KMeans filters outliers before clustering, resulting in more compact and representative cluster centroids.

---

## 🛠️ Installation & Usage

### 1. Clone the repository

    git clone https://github.com/khyati2107/HAR-Project.git
    cd HAR-Project

### 2. Create a virtual environment

    python -m venv myenv
    myenv\Scripts\activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Download the dataset
Download the UCI HAR Dataset from:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Extract it so your folder looks like:

    HAR-Project/
    └── UCI HAR Dataset/
        ├── train/
        └── test/

### 5. Run the pipeline

    python main.py --data_path "UCI HAR Dataset"
---

## 📦 Dependencies

    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    torch
    scipy
---

## 📈 Output Files

All figures are saved automatically to the `outputs/` folder:
- `pca_explained_variance.png`
- `dataset_overview.png`
- `clustering_results.png`
- `confusion_matrix_*.png` (one per model)
- `model_accuracy_comparison.png`
- `neural_network_training_loss.png`
- `neural_network_confusion_matrix.png`

---

## 👥 Authors

- **Khyati Prashant Jetly** — Dept. of Computer Science, BITS Pilani Dubai
- **Srinjita Roy Chowdhury** — Dept. of Computer Science, BITS Pilani Dubai
- **Soorya Kiran Kakkarayil** — Dept. of Computer Science, BITS Pilani Dubai

---

## 🏫 Institution

Birla Institute of Technology and Science, Pilani — Dubai Campus
