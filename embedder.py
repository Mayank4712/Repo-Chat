import os
import uuid
import logging

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

load_dotenv()
COLLECTION_NAME = "code_chunks"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
_model = None
logger = logging.getLogger("repo_chat")
BATCH_SIZE = 64


def _client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env.")
    try:
        return QdrantClient(url=url, api_key=api_key, timeout=30)
    except Exception as exc:
        raise RuntimeError("Could not configure Qdrant. Check QDRANT_URL and QDRANT_API_KEY.") from exc


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def setup_collection():
    client = _client()
    model = _get_model()
    try:
        if not client.collection_exists(COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=model.get_sentence_embedding_dimension(),
                    distance=Distance.COSINE,
                ),
            )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="repo_id",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
    except Exception as exc:
        logger.exception("Qdrant collection setup failed")
        raise RuntimeError(
            "Qdrant collection setup failed. Check QDRANT_URL, QDRANT_API_KEY, "
            "cluster reachability, and permissions."
        ) from exc


def index_chunks(chunks, repo_id=None):
    client = _client()
    model = _get_model()
    setup_collection()
    repo_id = repo_id or (chunks[0].get("repo_id") if chunks else None)
    if not repo_id:
        raise ValueError("A repository identifier is required for indexing.")
    try:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]),
            wait=True,
        )
        texts = [chunk["code"] for chunk in chunks]
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
        points = []
        for index, chunk in enumerate(chunks):
            payload = {**chunk, "repo_id": repo_id}
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{repo_id}:{chunk['file']}:{chunk['start_line']}:{chunk['end_line']}:{chunk['function']}"))
            points.append(PointStruct(id=point_id, vector=vectors[index].tolist(), payload=payload))
        for start in range(0, len(points), BATCH_SIZE):
            batch = points[start:start + BATCH_SIZE]
            client.upsert(collection_name=COLLECTION_NAME, points=batch, wait=True)
            logger.info("Indexed %d/%d chunks", min(start + len(batch), len(points)), len(points))
        return len(points)
    except Exception as exc:
        logger.exception("Qdrant indexing failed")
        raise RuntimeError(
            "Qdrant indexing failed. Check QDRANT_URL, QDRANT_API_KEY, "
            "cluster reachability, and permissions."
        ) from exc
