from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from train_model import ensemble_predict_proba, load_or_train, train, transform_feature_map
from vector_store import recommend, retrieve_context


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CAMERA_FEATURES = {"smile", "eye_open", "brow_tension", "mouth_open", "face_tilt"}
QUESTION_MINIMUM = 3


def get_questions() -> list[dict[str, Any]]:
    with (DATA_DIR / "question_bank.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def retrain_models() -> dict[str, Any]:
    artifact = train()
    return {
        "trained_at": artifact["trained_at"],
        "metrics": artifact["metrics"],
        "feature_importance": artifact["base_feature_importance"],
    }


def analyze_emotion(payload: dict[str, Any]) -> dict[str, Any]:
    artifact = load_or_train()
    answers = payload.get("answers") or {}
    user_note = (payload.get("note") or "").strip()
    images = _payload_images(payload)

    face_result = extract_face_features_from_images(images) if images else _no_face("No camera image supplied.")
    answer_result = answers_to_features(answers)
    feature_map = combine_features(artifact["feature_names"], face_result, answer_result)
    probabilities, probability_stats = predict_probabilities(artifact, feature_map)

    raw_top_confidence = probabilities[0]["score"]
    calibrated_confidence = calibrate_confidence(raw_top_confidence, probability_stats, face_result, answer_result)
    top_emotion = probabilities[0]["emotion"]

    context_text = build_context_text(top_emotion, feature_map, answers, user_note)
    rag_context = retrieve_context(top_emotion, context_text, top_k=4)
    recommendations = recommend(top_emotion, context_text, top_each=3)
    needs_questions = should_ask_questions(face_result, answers, calibrated_confidence, probability_stats)

    return {
        "emotion": top_emotion,
        "confidence": round(calibrated_confidence, 3),
        "rawConfidence": round(raw_top_confidence, 3),
        "probabilities": probabilities,
        "uncertainty": probability_stats,
        "needsQuestions": needs_questions,
        "questionReason": question_reason(face_result, answers, calibrated_confidence, probability_stats),
        "analysisQuality": analysis_quality(face_result, answer_result, calibrated_confidence, probability_stats),
        "face": face_result,
        "features": {name: round(float(value), 3) for name, value in feature_map.items()},
        "signals": summarize_signals(feature_map),
        "forestCluster": forest_cluster(artifact, feature_map),
        "model": {
            "trainedAt": artifact["trained_at"],
            "metrics": artifact["metrics"],
            "featureImportance": artifact["base_feature_importance"][:5],
            "modelWeights": artifact.get("model_weights", {}),
        },
        "ragContext": rag_context,
        "recommendations": recommendations,
        "safetyNote": "This is an assistive wellness prototype, not a diagnosis or mental-health assessment.",
    }


def extract_face_features_from_images(images: list[str]) -> dict[str, Any]:
    frame_results = [extract_face_features(image) for image in images[:8]]
    detected = [result for result in frame_results if result.get("detected")]
    if not detected:
        message = frame_results[-1]["message"] if frame_results else "No camera frame supplied."
        result = _no_face(message)
        result["frameCount"] = len(frame_results)
        result["detectedFrames"] = 0
        return result

    weights = np.array([max(0.05, float(result.get("confidence", 0.0))) for result in detected])
    weights = weights / weights.sum()
    feature_names = list(detected[0]["features"].keys())
    feature_matrix = np.array(
        [[float(result["features"][name]) for name in feature_names] for result in detected],
        dtype=float,
    )
    weighted_features = np.average(feature_matrix, axis=0, weights=weights)
    stability = 1.0 - min(1.0, float(np.mean(np.std(feature_matrix, axis=0))) * 2.2)
    best = max(detected, key=lambda result: result.get("confidence", 0.0))
    detection_ratio = len(detected) / max(1, len(frame_results))
    confidence = min(
        1.0,
        float(np.average([result["confidence"] for result in detected], weights=weights)) * 0.72
        + stability * 0.18
        + detection_ratio * 0.1,
    )

    cues = _average_cues(detected, weights)
    return {
        "detected": True,
        "confidence": round(float(confidence), 3),
        "message": "Multi-frame OpenCV scan completed.",
        "box": best.get("box"),
        "cues": {
            **cues,
            "frameCount": len(frame_results),
            "detectedFrames": len(detected),
            "stability": round(float(stability), 3),
        },
        "frameCount": len(frame_results),
        "detectedFrames": len(detected),
        "features": {
            name: float(value)
            for name, value in zip(feature_names, weighted_features)
        },
    }


def extract_face_features(image_data: str) -> dict[str, Any]:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover - depends on environment packages
        return _no_face(f"OpenCV is unavailable: {exc}")

    try:
        encoded = image_data.split(",", 1)[1] if "," in image_data else image_data
        raw = base64.b64decode(encoded)
        buffer = np.frombuffer(raw, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except Exception as exc:
        return _no_face(f"Could not decode image: {exc}")

    if image is None:
        return _no_face("Camera frame could not be read.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    face_cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"),
    ]
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = []
    for cascade in face_cascades:
        found = cascade.detectMultiScale(equalized, scaleFactor=1.1, minNeighbors=5, minSize=(76, 76))
        faces.extend(found)

    if len(faces) == 0:
        return _no_face("No face detected. Try more light, face the camera directly, or answer the questions.")

    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
    roi_gray = equalized[y : y + h, x : x + w]
    upper = roi_gray[: max(1, h // 2), :]
    lower = roi_gray[h // 2 :, :]

    eyes = eye_cascade.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=4, minSize=(18, 18))
    smiles_loose = smile_cascade.detectMultiScale(lower, scaleFactor=1.55, minNeighbors=12, minSize=(24, 14))
    smiles_strict = smile_cascade.detectMultiScale(lower, scaleFactor=1.65, minNeighbors=20, minSize=(24, 14))

    upper_edges = cv2.Canny(upper, 70, 165)
    lower_edges = cv2.Canny(lower, 70, 165)
    upper_edge_density = float(np.count_nonzero(upper_edges) / max(1, upper_edges.size))
    lower_edge_density = float(np.count_nonzero(lower_edges) / max(1, lower_edges.size))
    contrast = float(np.std(roi_gray) / 90.0)
    brightness = float(np.mean(gray[y : y + h, x : x + w]) / 255.0)
    sharpness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var() / 420.0)
    image_area = image.shape[0] * image.shape[1]
    face_area_ratio = (w * h) / max(1, image_area)

    smile_evidence = min(1.0, len(smiles_loose) * 0.28 + len(smiles_strict) * 0.42)
    smile = min(1.0, smile_evidence + lower_edge_density * 2.25 + contrast * 0.08)
    eye_open = min(1.0, len(eyes) / 2.0 + upper_edge_density * 0.35)
    brow_tension = min(1.0, upper_edge_density * 4.0 + max(0.0, contrast - 0.45) * 0.24)
    mouth_open = min(1.0, lower_edge_density * 4.2 + smile_evidence * 0.2)
    face_tilt = _estimate_face_tilt(eyes, w)

    lighting_quality = 1.0 - min(1.0, abs(brightness - 0.52) * 2.0)
    sharpness_quality = min(1.0, sharpness)
    detector_agreement = min(1.0, len(faces) / 2.0)
    confidence = min(
        1.0,
        0.2
        + min(0.22, face_area_ratio * 2.9)
        + (0.2 if len(eyes) >= 2 else 0.09 if len(eyes) == 1 else 0.0)
        + min(0.13, smile_evidence * 0.12)
        + lighting_quality * 0.12
        + sharpness_quality * 0.08
        + detector_agreement * 0.08,
    )

    return {
        "detected": True,
        "confidence": round(float(confidence), 3),
        "message": "Face detected with OpenCV Haar cascades.",
        "box": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
        "cues": {
            "faces": int(len(faces)),
            "eyes": int(len(eyes)),
            "smiles": int(len(smiles_loose)),
            "strictSmiles": int(len(smiles_strict)),
            "lighting": round(float(lighting_quality), 3),
            "contrast": round(float(min(1.0, contrast)), 3),
            "sharpness": round(float(sharpness_quality), 3),
            "faceCoverage": round(float(min(1.0, face_area_ratio * 4.0)), 3),
        },
        "features": {
            "smile": float(smile),
            "eye_open": float(eye_open),
            "brow_tension": float(brow_tension),
            "mouth_open": float(mouth_open),
            "face_tilt": float(face_tilt),
        },
    }


def answers_to_features(answers: dict[str, Any]) -> dict[str, Any]:
    questions = get_questions()
    weighted: dict[str, float] = {}
    weights: dict[str, float] = {}
    answered: dict[str, float] = {}

    for question in questions:
        question_id = question["id"]
        if question_id not in answers:
            continue
        try:
            value = min(5.0, max(1.0, float(answers[question_id])))
        except (TypeError, ValueError):
            continue

        normalized = (value - 1.0) / 4.0
        answered[question_id] = normalized
        for feature, weight in question["weights"].items():
            magnitude = abs(float(weight))
            target_value = normalized if weight >= 0 else 1.0 - normalized
            weighted[feature] = weighted.get(feature, 0.0) + target_value * magnitude
            weights[feature] = weights.get(feature, 0.0) + magnitude

    features = {feature: weighted[feature] / weights[feature] for feature in weighted}
    return {
        "features": features,
        "weights": weights,
        "answered": answered,
        "answeredCount": len(answered),
        "questionCount": len(questions),
    }


def combine_features(
    feature_names: list[str],
    face_result: dict[str, Any],
    answer_result: dict[str, Any],
) -> dict[str, float]:
    combined = {feature: 0.5 for feature in feature_names}
    answer_features = answer_result["features"]
    answer_strength = min(1.0, answer_result["answeredCount"] / max(1, answer_result.get("questionCount", 6)))

    if face_result.get("detected"):
        face_features = face_result.get("features", {})
        for feature, value in face_features.items():
            combined[feature] = float(value)

        inferred = infer_behavior_from_face(face_features)
        for feature, value in inferred.items():
            combined[feature] = float(value)

    face_confidence = float(face_result.get("confidence", 0.0))
    for feature, value in answer_features.items():
        if feature not in combined:
            continue

        answer_weight = 0.55 + 0.35 * answer_strength
        if feature in CAMERA_FEATURES and face_result.get("detected"):
            face_weight = max(0.08, face_confidence)
            combined[feature] = _weighted_average(
                [combined[feature], float(value)],
                [face_weight, answer_weight * 0.62],
            )
        else:
            prior_weight = 0.28 if face_result.get("detected") else 0.08
            combined[feature] = _weighted_average(
                [combined[feature], float(value)],
                [prior_weight, answer_weight],
            )

    combined["brow_tension"] = _clip(0.8 * combined["brow_tension"] + 0.2 * combined["stress"])
    combined["smile"] = _clip(0.84 * combined["smile"] + 0.16 * combined["self_valence"])
    combined["self_energy"] = _clip(0.72 * combined["self_energy"] + 0.16 * combined["eye_open"] + 0.12 * combined["mouth_open"])
    return combined


def infer_behavior_from_face(face_features: dict[str, float]) -> dict[str, float]:
    smile = float(face_features.get("smile", 0.5))
    eye_open = float(face_features.get("eye_open", 0.5))
    brow = float(face_features.get("brow_tension", 0.5))
    mouth = float(face_features.get("mouth_open", 0.5))
    tilt = float(face_features.get("face_tilt", 0.5))

    return {
        "self_valence": _clip(0.48 + (smile - brow) * 0.42 - tilt * 0.08),
        "self_energy": _clip(0.34 + eye_open * 0.32 + mouth * 0.25 + brow * 0.12),
        "social": _clip(0.35 + smile * 0.42 + (1.0 - brow) * 0.12),
        "stress": _clip(0.22 + brow * 0.38 + tilt * 0.18 + (1.0 - smile) * 0.12),
    }


def predict_probabilities(artifact: dict[str, Any], feature_map: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x_values = transform_feature_map(artifact, feature_map)
    labels = list(artifact["labels"])
    combined = ensemble_predict_proba(artifact, x_values)[0]

    ranked = sorted(zip(labels, combined), key=lambda item: item[1], reverse=True)
    top_score = float(ranked[0][1])
    second_score = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    entropy = _entropy(combined)
    stats = {
        "margin": round(top_score - second_score, 4),
        "entropy": round(entropy, 4),
        "agreement": round(1.0 - entropy, 4),
    }
    return [{"emotion": label, "score": round(float(score), 4)} for label, score in ranked], stats


def calibrate_confidence(
    raw_confidence: float,
    probability_stats: dict[str, Any],
    face_result: dict[str, Any],
    answer_result: dict[str, Any],
) -> float:
    face_quality = float(face_result.get("confidence", 0.0)) if face_result.get("detected") else 0.0
    answer_quality = min(1.0, answer_result["answeredCount"] / max(1, QUESTION_MINIMUM))
    evidence_quality = max(face_quality, answer_quality)
    margin = float(probability_stats["margin"])
    agreement = float(probability_stats["agreement"])
    confidence = raw_confidence * 0.62 + margin * 0.18 + agreement * 0.12 + evidence_quality * 0.08
    return _clip(confidence)


def forest_cluster(artifact: dict[str, Any], feature_map: dict[str, float]) -> dict[str, Any]:
    x_values = transform_feature_map(artifact, feature_map)
    leaves = artifact["random_forest"].apply(x_values)[0]
    weights = np.arange(1, len(leaves) + 1)
    cluster_id = int(np.sum(leaves * weights) % 997)
    return {
        "id": cluster_id,
        "method": "Random forest leaf-signature cluster",
        "description": "The forest divides similar feeling patterns by the leaf path reached across decision trees.",
    }


def should_ask_questions(
    face_result: dict[str, Any],
    answers: dict[str, Any],
    confidence: float,
    probability_stats: dict[str, Any],
) -> bool:
    answered_count = len([value for value in answers.values() if value is not None])
    if answered_count >= QUESTION_MINIMUM:
        return False
    if not face_result.get("detected"):
        return True
    if float(face_result.get("confidence", 0.0)) < 0.62:
        return True
    if float(probability_stats["margin"]) < 0.14:
        return True
    return confidence < 0.48


def question_reason(
    face_result: dict[str, Any],
    answers: dict[str, Any],
    confidence: float,
    probability_stats: dict[str, Any],
) -> str:
    if len([value for value in answers.values() if value is not None]) >= QUESTION_MINIMUM:
        return "Questionnaire signal included."
    if not face_result.get("detected"):
        return "The camera did not detect a face reliably, so questions improve the analysis."
    if float(face_result.get("confidence", 0.0)) < 0.62:
        return "The multi-frame facial signal is weak, so questions improve the analysis."
    if float(probability_stats["margin"]) < 0.14:
        return "Two emotions are close together, so questions help separate the pattern."
    if confidence < 0.48:
        return "The model is uncertain, so questions improve the analysis."
    return "Camera signal is strong enough for a first pass."


def analysis_quality(
    face_result: dict[str, Any],
    answer_result: dict[str, Any],
    confidence: float,
    probability_stats: dict[str, Any],
) -> dict[str, Any]:
    face_quality = float(face_result.get("confidence", 0.0)) if face_result.get("detected") else 0.0
    answer_quality = min(1.0, answer_result["answeredCount"] / max(1, answer_result.get("questionCount", 6)))
    overall = _clip(confidence * 0.62 + max(face_quality, answer_quality) * 0.26 + float(probability_stats["agreement"]) * 0.12)
    label = "high" if overall >= 0.72 else "medium" if overall >= 0.48 else "low"
    return {
        "label": label,
        "score": round(overall, 3),
        "faceQuality": round(face_quality, 3),
        "answerQuality": round(answer_quality, 3),
    }


def summarize_signals(feature_map: dict[str, float]) -> list[str]:
    signals = []
    valence = feature_map["self_valence"]
    stress = feature_map["stress"]
    energy = feature_map["self_energy"]
    smile = feature_map["smile"]
    brow = feature_map["brow_tension"]

    signals.append("positive valence" if valence >= 0.62 else "low valence" if valence <= 0.38 else "mixed valence")
    signals.append("high stress" if stress >= 0.66 else "low stress" if stress <= 0.34 else "moderate stress")
    signals.append("high energy" if energy >= 0.66 else "low energy" if energy <= 0.34 else "steady energy")
    if smile >= 0.62:
        signals.append("visible smile cue")
    if brow >= 0.66:
        signals.append("brow tension cue")
    return signals


def build_context_text(
    emotion: str,
    feature_map: dict[str, float],
    answers: dict[str, Any],
    user_note: str,
) -> str:
    high_features = [name for name, value in feature_map.items() if value >= 0.66]
    low_features = [name for name, value in feature_map.items() if value <= 0.34]
    answer_text = " ".join(f"{key}:{value}" for key, value in answers.items())
    return " ".join(
        [
            emotion,
            "high",
            " ".join(high_features),
            "low",
            " ".join(low_features),
            answer_text,
            user_note,
        ]
    )


def _payload_images(payload: dict[str, Any]) -> list[str]:
    images = payload.get("images")
    if isinstance(images, list):
        return [image for image in images if isinstance(image, str) and image][:8]
    image = payload.get("image")
    return [image] if isinstance(image, str) and image else []


def _average_cues(detected: list[dict[str, Any]], weights: np.ndarray) -> dict[str, Any]:
    numeric: dict[str, list[float]] = {}
    for result in detected:
        for key, value in result.get("cues", {}).items():
            if isinstance(value, (int, float)):
                numeric.setdefault(key, []).append(float(value))

    averaged = {}
    for key, values in numeric.items():
        if len(values) == len(weights):
            averaged[key] = round(float(np.average(values, weights=weights)), 3)
    return averaged


def _estimate_face_tilt(eyes: np.ndarray, face_width: int) -> float:
    if len(eyes) < 2:
        return 0.42
    sorted_eyes = sorted(eyes, key=lambda rect: rect[0])[:2]
    (_, y1, _, _), (_, y2, _, _) = sorted_eyes
    return min(1.0, abs(float(y1 - y2)) / max(1.0, face_width * 0.12))


def _entropy(probabilities: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    return entropy / max(1e-9, math.log(len(clipped)))


def _weighted_average(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0:
        return float(np.mean(values))
    return _clip(sum(value * weight for value, weight in zip(values, weights)) / total)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _no_face(message: str) -> dict[str, Any]:
    return {
        "detected": False,
        "confidence": 0.0,
        "message": message,
        "box": None,
        "cues": {},
        "frameCount": 0,
        "detectedFrames": 0,
        "features": {},
    }
