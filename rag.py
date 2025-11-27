from groq import Groq
from chroma_manager import ChromaManager
from dotenv import load_dotenv
import os

from query_pumbed import retrieve_from_chroma
load_dotenv()


def query_vector_store(user_question, k=5):
    """
    Retrive top-k relevent documents from chromadb for the given user question.


    """
    retrieved_docs = retrieve_from_chroma(user_question, k=k)
    return retrieved_docs

def build_prompt(context_docs, query):
    """
    Build a system prompt using context + the user query.

    """
    context_text ="\n\n".join(
        [f"Document {i+1}:\n{doc['content']}" for i, doc in enumerate(context_docs)]
    )

    prompt = f"""
You are a highly knowledgeable medical research assistanct.
Use ONly the following context to answer the question.
If the information is not available, say "The context does not contain enough information.



{context_text}

{query}
"""
    return prompt

def generate_answer(prompt):
    """
    Use Groq client with Llama 3 to generate an answer based on the prompt.
    """
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # Llama 3
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content

def rag_pipeline(user_question):
    print("Querying vector store...")
    docs = query_vector_store(user_question)
    if not docs:  # empty list or None
        print("⚠ No documents retrieved. Cannot build prompt.")
    return "The context does not contain enough information."

    print(f"Retrieved {len(docs)} documents.\n")


    print("Building prompt...")
    prompt = build_prompt(docs, user_question)

    print("Generating answer...")
    answer = generate_answer(prompt)

    return answer

if __name__ == "__main__":
    question = "What does research say about deep learning in medical imaging?"
    final_answer = rag_pipeline(question)

    print("\n\n=== FINAL ANSWER ===\n")
    print(final_answer)