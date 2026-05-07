from __future__ import annotations

import json
import warnings
from csv import DictReader
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vector_store import build_vector_store


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
MODEL_PATH = STORAGE_DIR / "emotion_model.joblib"
CUSTOM_DATA_PATH = DATA_DIR / "custom_emotion_samples.csv"
FER_DATA_PATHS = [
    DATA_DIR / "kaggle" / "fer2013.csv",
    DATA_DIR / "kaggle" / "icml_face_data.csv",
    DATA_DIR / "fer2013.csv",
    DATA_DIR / "icml_face_data.csv",
]
ARTIFACT_VERSION = 2


def load_pattern_config() -> dict[str, Any]:
    with (DATA_DIR / "emotion_patterns.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def generate_training_data(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    config = load_pattern_config()
    rng = np.random.default_rng(seed)

    features = config["feature_names"]
    label_centers = _centers_by_emotion(config)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    source_counts: dict[str, int] = {}

    for source in config["sources"]:
        source_name = source["name"]
        samples_per_emotion = int(source["samples_per_emotion"])
        spread = float(source["spread"])

        for pattern in source["patterns"]:
            center = np.array(pattern["center"], dtype=float)
            emotion = pattern["emotion"]

            generated = rng.normal(center, spread, size=(samples_per_emotion, len(features)))
            generated = _augment_source(generated, center, spread, rng)

            rows.extend(generated)
            labels.extend([emotion] * samples_per_emotion)
            source_counts[source_name] = source_counts.get(source_name, 0) + samples_per_emotion

    custom_x, custom_y = _load_custom_samples(features)
    if len(custom_y):
        rows.extend(custom_x)
        labels.extend(custom_y)
        source_counts["custom_emotion_samples"] = len(custom_y)

    fer_x, fer_y, fer_source = _load_fer2013_samples(features, label_centers)
    if len(fer_y):
        rows.extend(fer_x)
        labels.extend(fer_y)
        source_counts[fer_source] = len(fer_y)

    base_x = np.vstack(rows)
    y = np.array(labels)
    mirrored_x, mirrored_y = _make_boundary_samples(base_x, y, rng)
    return np.vstack([base_x, mirrored_x]), np.concatenate([y, mirrored_y]), features, source_counts


def engineer_features(base_x: np.ndarray, base_feature_names: list[str]) -> tuple[np.ndarray, list[str]]:
    index = {name: position for position, name in enumerate(base_feature_names)}

    def col(name: str) -> np.ndarray:
        return base_x[:, index[name]]

    smile = col("smile")
    eye_open = col("eye_open")
    brow_tension = col("brow_tension")
    mouth_open = col("mouth_open")
    face_tilt = col("face_tilt")
    valence = col("self_valence")
    energy = col("self_energy")
    social = col("social")
    stress = col("stress")

    derived = [
        smile * valence,
        brow_tension * stress,
        eye_open * mouth_open,
        (valence + social + (1.0 - stress)) / 3.0,
        (stress + brow_tension + eye_open + energy) / 4.0,
        ((1.0 - social) + (1.0 - energy) + (1.0 - valence)) / 3.0,
        np.clip((np.abs(smile - 0.5) + np.abs(mouth_open - 0.5) + np.abs(brow_tension - 0.5)) / 1.5, 0.0, 1.0),
        (smile + eye_open + mouth_open + brow_tension) / 4.0,
        (stress + energy + brow_tension + (1.0 - valence)) / 4.0,
        np.clip(np.abs(valence - 0.5) * 0.55 + np.abs(stress - 0.5) * 0.45 + face_tilt * 0.18, 0.0, 1.0),
        np.clip((smile - brow_tension + valence + (1.0 - stress)) / 3.0, 0.0, 1.0),
        np.clip((mouth_open + eye_open + (1.0 - face_tilt)) / 3.0, 0.0, 1.0),
    ]
    derived_names = [
        "smile_valence_match",
        "stress_brow_coupling",
        "eye_mouth_activation",
        "calm_balance",
        "threat_activation",
        "withdrawal_index",
        "expressive_intensity",
        "camera_expression_load",
        "negative_arousal",
        "mixed_state_index",
        "positive_regulation",
        "surprise_readiness",
    ]

    engineered = np.column_stack([base_x, *derived])
    return engineered, [*base_feature_names, *derived_names]


def train(seed: int = 42) -> dict[str, Any]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    base_x, y, base_feature_names, source_counts = generate_training_data(seed)
    x, model_feature_names = engineer_features(base_x, base_feature_names)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.2,
        random_state=seed + 7,
        stratify=y_train_full,
    )

    models = _make_models(seed)
    for model in models.values():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(x_train, y_train)

    validation_scores = {
        name: float(accuracy_score(y_val, model.predict(x_val)))
        for name, model in models.items()
    }
    model_weights = _weights_from_validation(validation_scores)

    test_predictions = {
        name: model.predict(x_test)
        for name, model in models.items()
    }
    ensemble_pred = _ensemble_predict(models, model_weights, x_test)
    labels = list(models["random_forest"].classes_)

    metrics = {
        "ensemble_accuracy": round(float(accuracy_score(y_test, ensemble_pred)), 4),
        "random_forest_accuracy": round(float(accuracy_score(y_test, test_predictions["random_forest"])), 4),
        "neural_network_accuracy": round(float(accuracy_score(y_test, test_predictions["neural_net"])), 4),
        "extra_trees_accuracy": round(float(accuracy_score(y_test, test_predictions["extra_trees"])), 4),
        "gradient_boost_accuracy": round(float(accuracy_score(y_test, test_predictions["gradient_boost"])), 4),
        "validation_scores": {name: round(value, 4) for name, value in validation_scores.items()},
        "model_weights": {name: round(value, 4) for name, value in model_weights.items()},
        "classification_report": classification_report(y_test, ensemble_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred, labels=labels).tolist(),
        "dataset_sources": source_counts,
        "sample_count": int(len(y)),
        "base_feature_count": len(base_feature_names),
        "engineered_feature_count": len(model_feature_names),
        "neural_network_shape": "3 hidden layers x 9 neurons",
        "feature_engineering": "12 interaction features added before the model stack",
    }

    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "feature_names": base_feature_names,
        "model_feature_names": model_feature_names,
        "labels": labels,
        "models": models,
        "model_weights": model_weights,
        "random_forest": models["random_forest"],
        "neural_net": models["neural_net"],
        "metrics": metrics,
        "feature_importance": _feature_importance(models["random_forest"], model_feature_names),
        "base_feature_importance": _base_feature_importance(models["random_forest"], base_feature_names, model_feature_names),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, MODEL_PATH)
    build_vector_store()
    return artifact


def load_or_train() -> dict[str, Any]:
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        if artifact.get("artifact_version") == ARTIFACT_VERSION:
            return artifact
    return train()


def transform_feature_map(artifact: dict[str, Any], feature_map: dict[str, float]) -> np.ndarray:
    base_names = artifact["feature_names"]
    base_x = np.array([[feature_map[name] for name in base_names]], dtype=float)
    engineered, _ = engineer_features(base_x, base_names)
    return engineered


def ensemble_predict_proba(artifact: dict[str, Any], x_values: np.ndarray) -> np.ndarray:
    models = artifact.get("models")
    if not models:
        rf_proba = artifact["random_forest"].predict_proba(x_values)
        nn_proba = artifact["neural_net"].predict_proba(x_values)
        return 0.58 * rf_proba + 0.42 * nn_proba

    weights = artifact["model_weights"]
    combined = np.zeros_like(next(iter(models.values())).predict_proba(x_values), dtype=float)
    for name, model in models.items():
        combined += weights[name] * model.predict_proba(x_values)
    return combined


def _make_models(seed: int) -> dict[str, Any]:
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=420,
            max_depth=14,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=360,
            max_depth=16,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=seed + 11,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.045,
            max_depth=3,
            subsample=0.9,
            random_state=seed + 23,
        ),
        "neural_net": Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(9, 9, 9),
                        activation="relu",
                        solver="adam",
                        alpha=0.0006,
                        learning_rate_init=0.003,
                        max_iter=900,
                        early_stopping=False,
                        random_state=seed + 37,
                    ),
                ),
            ]
        ),
    }


def _augment_source(
    generated: np.ndarray,
    center: np.ndarray,
    spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    distortion = rng.normal(0.0, spread / 3.0, size=(1, generated.shape[1]))
    generated = generated + distortion

    mask = rng.random(generated.shape) < 0.04
    fallback = rng.normal(center, spread * 1.35, size=generated.shape)
    generated = np.where(mask, fallback, generated)

    return np.clip(generated, 0.0, 1.0)


def _make_boundary_samples(
    base_x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    unique_labels = sorted(set(y))

    for emotion in unique_labels:
        current = base_x[y == emotion]
        others = base_x[y != emotion]
        sample_count = min(45, len(current))
        current_pick = current[rng.choice(len(current), size=sample_count, replace=False)]
        other_pick = others[rng.choice(len(others), size=sample_count, replace=False)]
        mixed = 0.88 * current_pick + 0.12 * other_pick
        mixed = np.clip(mixed + rng.normal(0.0, 0.024, size=mixed.shape), 0.0, 1.0)
        rows.append(mixed)
        labels.extend([emotion] * sample_count)

    return np.vstack(rows), np.array(labels)


def _load_custom_samples(feature_names: list[str]) -> tuple[list[np.ndarray], list[str]]:
    if not CUSTOM_DATA_PATH.exists():
        return [], []

    rows: list[np.ndarray] = []
    labels: list[str] = []
    with CUSTOM_DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        for row in reader:
            emotion = (row.get("emotion") or "").strip().lower()
            if not emotion:
                continue
            try:
                values = [min(1.0, max(0.0, float(row[name]))) for name in feature_names]
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(np.array(values, dtype=float))
            labels.append(emotion)
    return rows, labels


def _load_fer2013_samples(
    feature_names: list[str],
    label_centers: dict[str, np.ndarray],
    per_emotion_limit: int = 1200,
) -> tuple[list[np.ndarray], list[str], str]:
    data_path = next((path for path in FER_DATA_PATHS if path.exists()), None)
    if data_path is None:
        return [], [], "kaggle_fer2013"

    rows: list[np.ndarray] = []
    labels: list[str] = []
    counts: dict[str, int] = {}

    with data_path.open("r", encoding="utf-8", newline="") as handle:
        reader = DictReader(handle)
        for raw_row in reader:
            row = {key.strip(): value for key, value in raw_row.items() if key}
            emotion = _map_fer_label(row.get("emotion") or row.get("Emotion") or row.get("label") or "")
            pixels = row.get("pixels") or row.get(" Pixels") or row.get("pixel") or row.get("pixels ")
            if not emotion or not pixels or emotion not in label_centers:
                continue
            if counts.get(emotion, 0) >= per_emotion_limit:
                continue

            image_features = _features_from_fer_pixels(pixels)
            if not image_features:
                continue

            center = label_centers[emotion]
            blended = []
            for index, name in enumerate(feature_names):
                pixel_value = image_features.get(name, 0.5)
                prior_value = float(center[index])
                blended.append(_clip(0.58 * pixel_value + 0.42 * prior_value))

            rows.append(np.array(blended, dtype=float))
            labels.append(emotion)
            counts[emotion] = counts.get(emotion, 0) + 1

    return rows, labels, f"kaggle_fer2013:{data_path.name}"


def _features_from_fer_pixels(pixel_text: str) -> dict[str, float] | None:
    values = np.fromstring(pixel_text, sep=" ", dtype=float)
    if values.size != 48 * 48:
        return None

    image = values.reshape(48, 48) / 255.0
    gradient_y, gradient_x = np.gradient(image)
    edges = np.sqrt(gradient_x**2 + gradient_y**2)

    upper = image[:24, :]
    lower = image[24:, :]
    eye_band = image[11:23, 7:41]
    brow_band = image[7:17, 7:41]
    mouth_band = image[30:42, 11:37]
    left_edges = edges[:, :24]
    right_edges = edges[:, 24:]

    contrast = float(np.std(image))
    upper_edges = float(np.mean(edges[:24, :]))
    lower_edges = float(np.mean(edges[24:, :]))
    eye_contrast = float(np.std(eye_band))
    brow_edges = float(np.mean(edges[7:20, 7:41]))
    mouth_edges = float(np.mean(edges[30:42, 11:37]))
    mouth_darkness = 1.0 - float(np.mean(mouth_band))
    lower_brightness = float(np.mean(lower))
    upper_brightness = float(np.mean(upper))
    tilt = _clip(abs(float(np.mean(left_edges) - np.mean(right_edges))) * 8.0)

    smile = _clip(0.42 + (lower_brightness - upper_brightness) * 1.8 + mouth_edges * 3.5 - brow_edges * 0.9)
    eye_open = _clip(eye_contrast * 5.0 + upper_edges * 3.2 + contrast * 0.7)
    brow_tension = _clip(brow_edges * 6.4 + contrast * 0.5)
    mouth_open = _clip(mouth_edges * 6.8 + mouth_darkness * 0.22)
    face_tilt = tilt

    self_valence = _clip(0.48 + (smile - brow_tension) * 0.36 - face_tilt * 0.08)
    self_energy = _clip(0.32 + eye_open * 0.34 + mouth_open * 0.22 + brow_tension * 0.12)
    social = _clip(0.34 + smile * 0.42 + (1.0 - brow_tension) * 0.1)
    stress = _clip(0.2 + brow_tension * 0.42 + face_tilt * 0.16 + (1.0 - smile) * 0.1)

    return {
        "smile": smile,
        "eye_open": eye_open,
        "brow_tension": brow_tension,
        "mouth_open": mouth_open,
        "face_tilt": face_tilt,
        "self_valence": self_valence,
        "self_energy": self_energy,
        "social": social,
        "stress": stress,
    }


def _map_fer_label(value: str) -> str | None:
    normalized = str(value).strip().lower()
    label_map = {
        "0": "angry",
        "angry": "angry",
        "anger": "angry",
        "1": "angry",
        "disgust": "angry",
        "2": "fearful",
        "fear": "fearful",
        "fearful": "fearful",
        "3": "happy",
        "happy": "happy",
        "happiness": "happy",
        "4": "sad",
        "sad": "sad",
        "sadness": "sad",
        "5": "surprised",
        "surprise": "surprised",
        "surprised": "surprised",
        "6": "calm",
        "neutral": "calm",
        "calm": "calm",
    }
    return label_map.get(normalized)


def _centers_by_emotion(config: dict[str, Any]) -> dict[str, np.ndarray]:
    centers: dict[str, list[np.ndarray]] = {}
    for source in config["sources"]:
        for pattern in source["patterns"]:
            centers.setdefault(pattern["emotion"], []).append(np.array(pattern["center"], dtype=float))
    return {emotion: np.mean(values, axis=0) for emotion, values in centers.items()}


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _weights_from_validation(validation_scores: dict[str, float]) -> dict[str, float]:
    adjusted = {
        name: max(0.001, (score - 0.45) ** 2)
        for name, score in validation_scores.items()
    }
    total = sum(adjusted.values())
    return {name: value / total for name, value in adjusted.items()}


def _ensemble_predict(models: dict[str, Any], model_weights: dict[str, float], x_values: np.ndarray) -> np.ndarray:
    labels = list(models["random_forest"].classes_)
    combined = np.zeros_like(models["random_forest"].predict_proba(x_values), dtype=float)
    for name, model in models.items():
        combined += model_weights[name] * model.predict_proba(x_values)
    return np.array([labels[int(index)] for index in combined.argmax(axis=1)])


def _feature_importance(model: RandomForestClassifier, feature_names: list[str]) -> list[dict[str, Any]]:
    pairs = zip(feature_names, model.feature_importances_)
    ranked = sorted(pairs, key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": round(float(value), 4)} for name, value in ranked]


def _base_feature_importance(
    model: RandomForestClassifier,
    base_feature_names: list[str],
    model_feature_names: list[str],
) -> list[dict[str, Any]]:
    importance = dict(zip(model_feature_names, model.feature_importances_))
    totals = {name: float(importance.get(name, 0.0)) for name in base_feature_names}
    derived_links = {
        "smile": ["smile_valence_match", "camera_expression_load", "positive_regulation"],
        "eye_open": ["eye_mouth_activation", "threat_activation", "camera_expression_load", "surprise_readiness"],
        "brow_tension": ["stress_brow_coupling", "threat_activation", "expressive_intensity", "camera_expression_load"],
        "mouth_open": ["eye_mouth_activation", "expressive_intensity", "camera_expression_load", "surprise_readiness"],
        "face_tilt": ["mixed_state_index", "surprise_readiness"],
        "self_valence": ["smile_valence_match", "calm_balance", "withdrawal_index", "negative_arousal", "positive_regulation"],
        "self_energy": ["threat_activation", "withdrawal_index", "negative_arousal"],
        "social": ["calm_balance", "withdrawal_index"],
        "stress": ["stress_brow_coupling", "calm_balance", "threat_activation", "negative_arousal", "positive_regulation"],
    }
    for base_name, links in derived_links.items():
        totals[base_name] += sum(float(importance.get(link, 0.0)) for link in links) / max(1, len(links))

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [{"feature": name, "importance": round(float(value), 4)} for name, value in ranked]


if __name__ == "__main__":
    trained = train()
    print(json.dumps({"trained_at": trained["trained_at"], "metrics": trained["metrics"]}, indent=2))
