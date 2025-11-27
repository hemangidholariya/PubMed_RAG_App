import streamlit as st
from grpc.framework.interfaces.base.utilities import completion
from openai import api_key

from pubmed import PubMedRetriever
from chroma_manager import ChromaManager
from query_pumbed import retrieve_from_chroma
from groq import Groq
import os
from dotenv import load_dotenv

from ingest_pubmed import MAX_ARTICLES

load_dotenv()

MAX_ARTICLES = 300

def convert_abstract_dict_to_text(abstract_dict):
    text_parts = []
    for label, content in abstract_dict.items():
        text_parts.append(f"{label}: {content}")
    return "\n".join(text_parts)

def ingest_pubmed(search_term):
    st.info(f"Searching PubMed for: {search_term}")
    pmids = PubMedRetriever.search_pubmed_articles(
        search_term, max_results=MAX_ARTICLES
    )
    st.success(f"Retrived {len(pmids)} PMIDs.")

    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    st.info(f"Ingesting {len(articles)} articles into chromadb...")

    manager = ChromaManager(
        persist_directory="pubmed_vector_db"
    )
    manager.create_collection(name="pubmed_collection")

    for article in articles:
        abstract_text = convert_abstract_dict_to_text(
            article["abstract"]
        )
        content = f"""
TITLE: {article["title"]}

ABSTRACT:
{abstract_text}

JOURNAL: {article["journal"]}
AUTHOR: {article["author"]}
YEAR: {article["publication_date"]}
"""

        manager.ingest_document(
            doc_id=article["pmid"],
            text=content,
            metadata={
                "pmid": article["pmid"],
                "journal": article["journal"],
                "year": article["publication_date"]
            }

        )
    st.success("Ingesting complete!")
    return manager

def build_prompt(context_docs, query):
    context_text = "\n\n".join([f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(context_docs)])
    prompt = f"""
You are a knowledgeable medical research assistant.
Use ONLY the following context to answer the question.
If the information is not available, say "The context does not contain enough information.

{context_text}
    
Question: {query}
"""
    return prompt

def generate_answer(prompt):
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages = [{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content

st.title("PubMed RAG App")
st.sidebar.header("Document Search & Ingestion")

search_term = st.sidebar.text_input("Search PubMed", value=" Diabetes ")

if st.sidebar.button("Ingest Articles"):
    if search_term.strip() == "":
        st.sidebar.error("Please enter a search term.")
    else:
        manager = ingest_pubmed(search_term)

st.header("Ask a Question")
user_question = st.text_input("Type your question here")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        st.info("Querying vector store...")
        docs = retrieve_from_chroma(user_question, k=10)

        if not docs:
            st.warning("No results found.")
        else:
            st.success(f"Retrieved {len(docs)} documents.")
            prompt = build_prompt(docs, user_question)

            st.info("Generating Answer...")
            answer = generate_answer(prompt)
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved Documents (Preview)")
            for i, doc in enumerate(docs):
                st.markdown(f"**Document {i+1}:**")
                st.text(doc["content"][:500] + "...")