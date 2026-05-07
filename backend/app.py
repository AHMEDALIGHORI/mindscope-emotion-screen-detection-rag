from __future__ import annotations

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from emotion_engine import analyze_emotion, get_questions, retrain_models
from train_model import load_or_train
from vector_store import search


app = Flask(__name__)
CORS(app)


@app.get("/api/health")
def health():
    artifact = load_or_train()
    return jsonify(
        {
            "status": "ok",
            "modelTrainedAt": artifact["trained_at"],
            "metrics": artifact["metrics"],
            "features": artifact["feature_names"],
            "modelFeatures": artifact.get("model_feature_names", artifact["feature_names"]),
            "featureImportance": artifact.get("base_feature_importance", artifact["feature_importance"])[:5],
            "modelWeights": artifact.get("model_weights", {}),
        }
    )


@app.get("/api/questions")
def questions():
    return jsonify({"questions": get_questions()})


@app.post("/api/analyze")
def analyze():
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify(analyze_emotion(payload))


@app.post("/api/train")
def train():
    return jsonify(retrain_models())


@app.post("/api/rag/search")
def rag_search():
    payload = request.get_json(force=True, silent=True) or {}
    query = payload.get("query", "")
    top_k = int(payload.get("topK", 6))
    return jsonify({"results": search(query, top_k=top_k)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=os.getenv("FLASK_DEBUG") == "1")
