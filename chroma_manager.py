import chromadb
from chromadb.utils import embedding_functions

class ChromaManager:
    def __init__(self, persist_directory="pubmed_vector_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)

        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def create_collection(self,name="pubmed_collection"):
        self.collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_fn)
        return self.collection

    def ingest_document(self,doc_id, text, metadata=None):
        self.collection.add(
            ids=[doc_id],
            documents=[text],
            metadatas=[metadata or {}]
        )

    def query(self, query_text, k=5):
        result = self.collection.query(
            query_texts = [query_text],
            n_results=k

        )
        return result
