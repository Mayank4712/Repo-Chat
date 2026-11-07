from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
client = QdrantClient(url="http://localhost:6333")

COLLECTION_NAME = "code_chunks"

def search(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )
    return results.points


if __name__ == "__main__":
    query = "how does login work"
    results = search(query)

    for point in results:
        print(f"Score: {point.score:.4f}")
        print(f"File: {point.payload['file']}")
        print(f"Function: {point.payload['function']}")
        print(f"Code:\n{point.payload['code']}")
        print("-" * 40)