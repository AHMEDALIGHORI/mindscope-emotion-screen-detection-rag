from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
VECTOR_PATH = STORAGE_DIR / "vector_store.joblib"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_documents() -> list[dict[str, Any]]:
    knowledge = _load_json(DATA_DIR / "rag_knowledge.json")
    book_knowledge = _load_optional_json(DATA_DIR / "emotion_books_knowledge.json")
    recommendations = _load_json(DATA_DIR / "recommendations.json")["items"]

    docs: list[dict[str, Any]] = []
    for item in [*knowledge, *book_knowledge]:
        docs.append(
            {
                "id": item["id"],
                "kind": "knowledge",
                "title": item["title"],
                "emotion_targets": item.get("emotion_targets", []),
                "text": " ".join(
                    [
                        item["title"],
                        " ".join(item.get("emotion_targets", [])),
                        item["text"],
                    ]
                ),
                "payload": item,
            }
        )

    for index, item in enumerate(recommendations):
        docs.append(
            {
                "id": f"recommendation-{index}",
                "kind": "recommendation",
                "title": item["title"],
                "emotion_targets": item.get("emotion_targets", []),
                "text": " ".join(
                    [
                        item["type"],
                        item["title"],
                        item.get("tone", ""),
                        item.get("description", ""),
                        " ".join(item.get("emotion_targets", [])),
                        " ".join(item.get("tags", [])),
                    ]
                ),
                "payload": item,
            }
        )
    return docs


def _load_optional_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _load_json(path)


def build_vector_store() -> dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    documents = _build_documents()
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform([doc["text"] for doc in documents])
    artifact = {
        "documents": documents,
        "vectorizer": vectorizer,
        "matrix": matrix,
    }
    joblib.dump(artifact, VECTOR_PATH)
    return artifact


def load_vector_store() -> dict[str, Any]:
    if not VECTOR_PATH.exists():
        return build_vector_store()
    return joblib.load(VECTOR_PATH)


def search(query: str, top_k: int = 6) -> list[dict[str, Any]]:
    artifact = load_vector_store()
    query_vector = artifact["vectorizer"].transform([query])
    scores = cosine_similarity(query_vector, artifact["matrix"])[0]
    ranked_indexes = scores.argsort()[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for index in ranked_indexes:
        doc = artifact["documents"][int(index)]
        results.append(
            {
                "score": float(scores[index]),
                "kind": doc["kind"],
                "title": doc["title"],
                "emotion_targets": doc["emotion_targets"],
                "payload": doc["payload"],
            }
        )
    return results


def recommend(emotion: str, context: str, top_each: int = 3) -> dict[str, list[dict[str, Any]]]:
    query = f"{emotion} {context}"
    matches = search(query, top_k=24)
    grouped: dict[str, list[dict[str, Any]]] = {"movie": [], "book": [], "joke": []}

    for match in matches:
        if match["kind"] != "recommendation":
            continue
        item = match["payload"]
        item_type = item["type"]
        if item_type not in grouped or len(grouped[item_type]) >= top_each:
            continue

        target_bonus = 0.12 if emotion in item.get("emotion_targets", []) else 0.0
        grouped[item_type].append(
            {
                **item,
                "score": round(min(1.0, match["score"] + target_bonus), 3),
                "why": _why_recommended(emotion, item),
            }
        )

    return grouped


def retrieve_context(emotion: str, context: str, top_k: int = 3) -> list[dict[str, Any]]:
    query = f"{emotion} {context}"
    return [item for item in search(query, top_k=top_k * 4) if item["kind"] == "knowledge"][:top_k]


def _why_recommended(emotion: str, item: dict[str, Any]) -> str:
    if emotion in item.get("emotion_targets", []):
        return f"Matched to {emotion} because its tone is {item.get('tone', 'supportive')}."
    return f"Nearby match from the vector store based on tone, tags and the current feeling profile."
