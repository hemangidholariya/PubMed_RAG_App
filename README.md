# PubMed RAG App 🧠🔬  
A Retrieval-Augmented Generation (RAG) application that fetches PubMed research articles, stores them in ChromaDB, and uses Groq LLaMA-3.3-70b-versatile to answer medical research questions based on retrieved context.

---

## 🚀 Features

- 🔍 Search PubMed articles using keywords  
- 📥 Ingest up to 300 articles into ChromaDB  
- 🔎 Retrieve relevant articles using vector similarity search  
- 🤖 Generate accurate responses using LLaMA-3.3 (Groq API)  
- 🧵 Streamlit UI with easy interaction  

---

## 🗂 Project Structure


│── app.py

│── pubmed.py

│── chroma_manager.py

│── ingest_pubmed.py

│── __init__.py

│── query_pumbed.py

│── requirements.txt

│── README.md

│── pubmed_vector_db/ (auto-created)

## 🛠 Installation

1. git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. python -m venv venv
3. pip install -r requirements.txt
4. Create a .env file:-
GROQ_API_KEY=your_api_key_here
5. streamlit run app.py

## 📌 How It Works
1. Search & Ingest

The app retrieves PubMed articles using PubMedRetriever, converts abstracts, and ingests them into ChromaDB.

2. Store as Embeddings

ChromaDB + SentenceTransformers creates embeddings for each article.

3. Query

User’s question goes to vector search → retrieves context.

4. Groq model Generates Answer

The model answers only using the retrieved scientific context.

## 🧪 Example Query
What does the latest research say about diabetes treatments?