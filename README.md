# HER2FusionNet: Multi-Head Cross-Attention Fusion for HER2 Classification

This repository contains the official implementation of **HER2FusionNet**, a deep learning framework designed for the 4-class classification of HER2 status (HER2-0, HER2-1+, HER2-2+, and HER2-3+) from histopathology image patches.

The architecture utilizes a dual-branch feature extraction system (Tissue-Net and Texture-Net) integrated through a 3-head cross-attention mechanism and a class-specific multi-head classification layer.

---

## 🔬 Model Architecture

The model consists of the following primary components:

1. **Tissue-Net:** Extracts structural and morphological features from 128×128 input patches.

2. **Texture-Net:** Utilizes learnable Gabor convolutional layers to capture fine-grained texture details from 256×256 input patches.

3. **Cross-Attention Fusion:** Aligns and fuses features from both branches using a multi-head attention mechanism.

4. **Class-Specific Multi-Heads:** Four independent classification heads specialized in identifying individual HER2 categories to reduce inter-class confusion.

---

## 📂 Dataset Structure

The code is designed to work with the BCI dataset and tested on BCNB dataset.

Ensure your data is organized in the following directory structure:

```text
Data/
├── train/
│   ├── HER2_0/
│   ├── HER2_1/
│   ├── HER2_2/
│   └── HER2_3/
│
├── validate/
│   ├── HER2_0/
│   ├── HER2_1/
│   ├── HER2_2/
│   └── HER2_3/
│
└── test/
    ├── HER2_0/
    ├── HER2_1/
    ├── HER2_2/
    └── HER2_3/
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/YOUR_USERNAME/HER2FusionNet.git
cd HER2FusionNet
pip install -r requirements.txt
```

---

### 2. Preprocessing

To extract patches and remove blank background patches from the raw BCI images, run:

```bash
python patch_generator.py
```

---

### 3. Training

To train the model using the class-specific multi-head architecture:

```bash
python her2_full_pipeline.py
```

The script will automatically save the best-performing weights as:

```text
best_her2_model.pth
```

based on validation accuracy.

---

### 4. Evaluation

To evaluate the model on the test set and generate the classification report and confusion matrix:

```bash
python evaluate_model.py
```

---

## 📊 Results

The evaluation script generates:

- **Classification Report**
  - Precision
  - Recall
  - F1-Score for each HER2 class

- **Confusion Matrix**
  - Visual representation of model performance across all four categories
  - Saved as:

```text
test_confusion_matrix.png
```

---

## 📄 Citation

If you use this code in your research, please cite our paper:

```text
[Insert your full paper citation here once published]
```

---

## 📌 Notes

1. Replace `YOUR_USERNAME` in the installation section with your actual GitHub username.

2. Once your paper is published, update the Citation section.

3. Ensure dataset usage complies with the original dataset license and usage policies.
