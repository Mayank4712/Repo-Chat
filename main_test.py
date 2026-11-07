from rank_bm25 import BM25Okapi
from embedder import COLLECTION_NAME, _client, _get_model
from qdrant_client.models import FieldCondition, Filter, MatchValue


def get_all_chunks(repo_id=None):
    client = _client()
    points = []
    offset = None
    query_filter = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]) if repo_id else None
    while True:
        batch, offset = client.scroll(collection_name=COLLECTION_NAME, limit=256, offset=offset, with_payload=True, scroll_filter=query_filter)
        points.extend(batch)
        if offset is None:
            return points


def semantic_search(query, top_k=10, repo_id=None):
    client = _client()
    model = _get_model()
    query_filter = Filter(must=[FieldCondition(key="repo_id", match=MatchValue(value=repo_id))]) if repo_id else None
    results = client.query_points(collection_name=COLLECTION_NAME, query=model.encode(query).tolist(), query_filter=query_filter, limit=top_k)
    return [(point.payload, point.score) for point in results.points]


def keyword_search(query, top_k=10, repo_id=None):
    all_points = get_all_chunks(repo_id)
    if not all_points:
        return []
    scores = BM25Okapi([point.payload.get("code", "").lower().split() for point in all_points]).get_scores(query.lower().split())
    ranked = sorted(zip(all_points, scores), key=lambda item: item[1], reverse=True)
    return [(point.payload, score) for point, score in ranked[:top_k]]


def reciprocal_rank_fusion(semantic_results, keyword_results, k=60):
    scores, lookup = {}, {}
    for rank, (payload, _) in enumerate(semantic_results + keyword_results):
        key = (payload["file"], payload["start_line"], payload["end_line"])
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        lookup[key] = payload
    return [{**lookup[key], "score": round(score, 4)} for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)]


def _is_broad_question(query):
    normalized = query.lower()
    return any(
        phrase in normalized
        for phrase in (
            "tell me about",
            "overview",
            "what is this project",
            "describe this project",
            "what does this project do",
        )
    )


def _overview_chunks(repo_id, limit=4):
    points = get_all_chunks(repo_id)
    readme = [
        point.payload
        for point in points
        if point.payload.get("file", "").lower() == "readme.md"
    ]
    top_level = [
        point.payload
        for point in points
        if "/" not in point.payload.get("file", "").replace("\\", "/")
        and point.payload.get("file", "").lower() != "readme.md"
    ]
    return (readme[:1] + top_level[: max(0, limit - 1)])[:limit]


def search(query, top_k=3, repo_id=None):
    ranked = reciprocal_rank_fusion(
        semantic_search(query, top_k=10, repo_id=repo_id),
        keyword_search(query, top_k=10, repo_id=repo_id),
    )[:top_k]
    if _is_broad_question(query):
        existing = {(item["file"], item["start_line"], item["end_line"]) for item in ranked}
        overview_results = []
        for item in _overview_chunks(repo_id):
            key = (item["file"], item["start_line"], item["end_line"])
            if key not in existing:
                overview_results.append({**item, "score": 0.0})
        ranked = overview_results + ranked
    return ranked[: max(top_k, 4) if _is_broad_question(query) else top_k]
