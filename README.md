# 🛢️ OilSpill-AI

**OilSpill-AI** is a deep learning project originally developed as the final exam for the *CT-213 – Artificial Intelligence for Mobile Robots* course at the Aeronautics Institute of Technology (ITA), taught by Professor Marcos Máximo.

The goal of this project is to develop an AI model capable of **segmenting ocean surface images** into four categories:

- 🌊 **Normal Water**  
- 🛢️ **Oil Spills**  
- 🐟 **Other Objects** (e.g., boats, debris)  
- 🏖️ **Background**

The segmentation is performed using a **U-Net** architecture, a popular convolutional neural network model for semantic segmentation tasks.

---

## 🚀 Features

- Image segmentation using U-Net  
- Dataset exploration and visualization tools  
- Training pipeline with model export (`.h5` weights + `.json` architecture)  
- Evaluation script for comparing predictions with ground truth  

---

## 📂 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/OilSpill-AI.git
cd OilSpill-AI
```

### 2. Download the Dataset

> ⚠️ The dataset used for training can be downloaded from [**LINK**].  
> (Replace `[LINK]` with the actual dataset URL.)

- After downloading, extract the dataset **into the root directory** of the cloned repository.

---

## 🔍 Explore the Dataset

To visualize and explore the dataset:

```bash
python explore_dataset.py
```

This script will display sample input images and their corresponding segmentation masks.

---

## 🧠 Train the U-Net Model

To train the model on your local machine:

```bash
python train_U_Net.py
```

This will:

- Train a U-Net model on the dataset  
- Save the model architecture to `model.json`  
- Save the trained weights to `model_weights.h5`

---

## 📈 Evaluate the Model

After training, evaluate the model's performance:

```bash
python evaluate_U_Net.py
```

This will compare your model’s predictions with the ground truth and display example results.

