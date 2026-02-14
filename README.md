# 🎬 IMDB Movie Review Sentiment Analysis using RNN

<p align="center">
  <img src="https://img.shields.io/badge/Python-NLP-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/TensorFlow-RNN-orange?style=for-the-badge&logo=tensorflow">
  <img src="https://img.shields.io/badge/NLP-Sentiment%20Analysis-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Project-DeepLearning%20Portfolio-black?style=for-the-badge">
</p>

<p align="center">
🚀 A Deep Learning based <b>IMDB Movie Review Sentiment Analyzer</b> built using Recurrent Neural Networks (RNN).  
This project demonstrates Natural Language Processing, sequence modeling, and deployment-ready AI workflows.
</p>

---

## 📌 Table of Contents

* [✨ Project Overview](#-project-overview)
* [🧠 Model Architecture](#-model-architecture)
* [🚀 Features](#-features)
* [⚙️ Tech Stack](#️-tech-stack)
* [📂 Project Structure](#-project-structure)
* [📊 Dataset](#-dataset)
* [🧪 Model Training](#-model-training)
* [🔮 Prediction Workflow](#-prediction-workflow)
* [📈 Results](#-results)
* [⚡ Installation](#️-installation)
* [▶️ How to Run](#️-how-to-run)
* [📬 Author](#-author)

---

# ✨ Project Overview

This project performs **sentiment classification** on movie reviews using a Deep Learning RNN model.

The system predicts whether a review is:

✅ Positive
❌ Negative

It showcases real-world NLP pipeline development — from tokenization and sequence padding to model deployment.

---

# 🧠 Model Architecture

The model uses a Recurrent Neural Network designed for sequence data:

* Word Index Encoding
* Sequence Padding
* Embedding Layer
* Simple RNN Layer
* Dense Output Layer (Sigmoid)

Pipeline:

```
Text Review → Tokenization → Sequence Padding → RNN → Sentiment Prediction
```

---

# 🚀 Features

* 🎬 NLP-based movie review analysis
* 🧠 Deep Learning with Simple RNN
* 📄 Text preprocessing pipeline
* ⚡ Streamlit-ready prediction interface
* 📊 Confidence score output
* 🧩 Clean modular project structure

---

# ⚙️ Tech Stack

| Technology         | Purpose              |
| ------------------ | -------------------- |
| Python             | Core Programming     |
| TensorFlow / Keras | RNN Model            |
| NumPy              | Numerical Processing |
| IMDB Dataset       | Training Data        |
| Streamlit          | Deployment UI        |

---

# 📂 Project Structure

```
IMDB-Review-RNN
│
├── main.py                 # Streamlit / prediction logic
├── embedding.ipynb         # Embedding experiments
├── simplernn.ipynb         # Model training notebook
├── prediction.ipynb        # Prediction workflow
├── simple_rnn_imdb.h5      # Trained RNN model
├── requirements.txt
└── README.md
```

---

# 📊 Dataset

Dataset used:

```
IMDB Movie Review Dataset (TensorFlow Keras)
```

Features:

* 50,000 labeled movie reviews
* Binary sentiment classification
* Pre-tokenized vocabulary

---

# 🧪 Model Training

Training includes:

* Tokenization using IMDB word index
* Sequence padding (maxlen = 500)
* RNN-based sequence learning
* Binary classification using sigmoid output

Key Parameters:

```
Max Sequence Length: 500
Model Type: Simple RNN
Loss: Binary Crossentropy
Optimizer: Adam
```

---

# 🔮 Prediction Workflow

1️⃣ User inputs movie review
2️⃣ Text is encoded using word index
3️⃣ Sequences padded to fixed length
4️⃣ Model predicts sentiment probability
5️⃣ Output displayed with confidence score

Example Output:

```
Sentiment: Positive
Prediction Score: 0.91
```

---

# 📈 Results

The RNN model successfully captures contextual sentiment patterns in movie reviews.

Visualizations included:

* Training Loss Curve
* Validation Accuracy Curve

---

# ⚡ Installation

Clone the repository:

```
git clone https://github.com/Vashishtha05/IMDB-Review-RNN.git
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ How to Run

Run the Streamlit app:

```
streamlit run main.py
```

---

# 📬 Author

**Vashishtha Verma**
AI/ML Engineer • Deep Learning & GenAI Enthusiast

Building intelligent systems using:

* Machine Learning
* Deep Learning (RNN, CNN, ANN)
* Full-Stack AI Development
* Strong DSA Foundations

