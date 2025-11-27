from pymed import article

# ingest_pubmed.py

from pubmed import PubMedRetriever
from chroma_manager import ChromaManager

MAX_ARTICLES = 300


def convert_abstract_dict_to_text(abstract_dict):

    text_parts = []
    for label, content in abstract_dict.items():
        text_parts.append(f"{label}: {content}")
    return "\n".join(text_parts)


def ingest_pubmed(search_term, max_articles=MAX_ARTICLES):
    print(f"🔍 Searching PubMed for: {search_term}")

    # -----------------------------
    # 1. Retrieve PMIDs
    # -----------------------------
    pmids = PubMedRetriever.search_pubmed_articles(search_term, max_results=MAX_ARTICLES)
    print(f"📗 Retrieved {len(pmids)} PMIDs")

    # -----------------------------
    # 2. Retrieve Full Articles
    # -----------------------------
    print("📥 Fetching full article abstracts...")
    articles = PubMedRetriever.fetch_pubmed_abstracts(pmids)
    print(f"📥 Fetched {len(articles)} articles")

    # -----------------------------
    # 3. Create Vector Collection
    # -----------------------------
    manager = ChromaManager()
    manager.create_collection()

    print("Documents in collection:", manager.collection.count())

    print("🧠 Ingesting into ChromaDB...")
    for idx, article in enumerate(articles, start=1):
        print(f" → Ingesting {idx}/{len(articles)} (PMID: {article['pmid']})")

        abstract_text = convert_abstract_dict_to_text(article["abstract"])

        # Merge title + abstract
        content = f"""
TITLE: {article['title']}

ABSTRACT:
{abstract_text}

JOURNAL: {article['journal']}
AUTHORS: {article['authors']}
YEAR: {article['publication_date']}
        """

        # Store in Chroma
        manager.ingest_document(
            doc_id=article["pmid"],
            text=content,
            metadata={
                "pmid": article["pmid"],
                "journal": article["journal"],
                "year": article["publication_date"]
            }
        )

    print("✅ Ingestion Complete!")
    return manager


if __name__ == "__main__":
    ingest_pubmed(" Diabetes", max_articles=MAX_ARTICLES)
