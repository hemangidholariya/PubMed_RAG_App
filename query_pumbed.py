from chroma_manager import ChromaManager

def retrieve_from_chroma(query_text, k=5):
    manager = ChromaManager(persist_directory="pubmed_vector_db")
    manager.create_collection(name="pubmed_collection")

    results = manager.query(query_text, k=k)
    if not results or len(results["documents"][0]) == 0:
        return []

    docs = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    for doc, meta in zip(documents, metadatas):
        docs.append({
            "content": doc,
            "metadata": meta
        })

    return docs


if __name__ == "__main__":
    output = retrieve_from_chroma("What does research say about applying deep learning in medical imaging?")
    print("Retrieved:", len(output), "documents")
    for d in output[:3]:
        print("\nPMID:", d["metadata"].get("pmid"))
        print(d["content"][:300], "...")