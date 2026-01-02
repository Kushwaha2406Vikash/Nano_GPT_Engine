# 🧠 Mini Decoder-Transformer LLM — Custom Healthcare QA Model

A **from-scratch implementation of a Mini Large Language Model (LLM)** built using **PyTorch + Decoder-Only Transformer Architecture**.  
This project focuses on **understanding and implementing the core mechanics behind word-level next-token prediction**, while training on **custom Women’s Healthcare Question-Answer data** (Pregnancy, Menstruation, Wellness, etc.).

It also includes a **FastAPI-powered REST API** so users can interact with the trained model in real-time.

---

## 🚀 Project Objectives

✔ Implement a **decoder-only transformer architecture** (GPT-style models)  
✔ Train on **custom QA healthcare dataset**  
✔ Understand **how LLMs predict the next word**  
✔ Deploy the trained model through an **API endpoint**  
✔ Build every component yourself — tokenizer, model, training loop & inference

---

## 🏗️ Architecture Overview

This project follows a **Decoder Transformer pipeline**:

1️⃣ Text Tokenization — using SentencePiece  
2️⃣ Token & Positional Embeddings  
3️⃣ Multi-Head Self-Attention  
4️⃣ Feed Forward Layers  
5️⃣ Output Projection  
6️⃣ Next-Token Prediction  
7️⃣ Training on QA dataset  
8️⃣ Real-time inference via API

The goal is **clarity + learning**, not complexity.

---

## 📁 Project Structure

LLM/
│
├── api.py # FastAPI Endpoint
├── inference_model.py # Inference Script
├── train.py # Training Script
├── transformer_block.py # Decoder Transformer Implementation
├── tokenizer.model # SentencePiece Model
├── tokenizer.vocab # Token Vocabulary
├── tinygpt.pt # Trained Model Weights
├── data.txt # Custom QA Healthcare Training Data
├── requirements.txt # Dependencies
├── .gitignore
│
├── .venv/ # Virtual Environment
├── pycache/ # Python Cache
└── .vscode/ # IDE Settings



## 🧪 Dataset — Women’s Healthcare QA

Training data is **custom & domain-focused**, including:

- Pregnancy
- Menstrual health
- General women’s wellness


---

## 🧬 Tech Stack

### Core ML
- torch
- numpy
- sentencepiece

### API Layer
- fastapi  
- uvicorn  

- pydantic  

### Additional Libraries (from requirements.txt) 

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt

## 2️⃣ Train the Model
- python train.py

### This will generate:

- tinygpt.pt

## 3️⃣ Run Inference 
- python inference_model.py  
### 4️⃣ Start API Server
- uvicorn api.py:app --reload
## Open in Browser:

http://127.0.0.1:8000/docs
