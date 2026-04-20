# customer-query-answering-nlp
Customer Query Answering System using NLP, Semantic Search, FAISS &amp; Transformers (DistilBERT)
🧠 Customer Query Answering System
🚀 Overview

This project is an NLP-based Question Answering system that uses Semantic Search + Transformer models to provide accurate answers to user queries.

It solves the problem of keyword-based search by understanding the meaning of queries.

🛠️ Tech Stack

Python
PyTorch
FAISS
Sentence Transformers
DistilBERT

🏗️ Architecture
Two-stage pipeline:
1. Semantic Search
Convert context → embeddings
Store in FAISS
Retrieve most relevant context
2. Answer Extraction
Use DistilBERT QA model
Predict answer span

📊 Dataset
Stanford Question Answering Dataset (SQuAD)
Context-based QA from Wikipedia

▶️ How to Run
pip install -r requirements.txt
python src/main.py

💡 Example

Input:
What is CLS?
Output:
A legal instrument resulting from a diplomatic compromise

⚠️ Limitations
Depends on dataset
No multi-context reasoning
Occasional incorrect outputs

🔮 Future Work
Top-K retrieval
Confidence score
Streamlit UI
LLM integration

👩‍💻 Author
Disha Sattesa
