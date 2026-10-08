"""Exploratory local baselines. Scores are not calibrated moderation probabilities."""

from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
LABELS_PATH = Path(os.getenv("MODERATION_LABELS_PATH", str(ROOT / "data" / "moderation_labels.jsonl")))
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RULES = ("off_platform", "restricted_item", "category_mismatch", "condition_contradiction")

# Contrastive examples are public policy descriptions, not training labels or a test set.
PROTOTYPES = {
    "off_platform": {
        "yes": [
            "Contact me on WhatsApp and pay outside the marketplace for a discount.",
            "Chat saya langsung lewat nomor telepon dan transfer di luar aplikasi.",
            "Use my email to arrange payment outside this platform.",
        ],
        "no": [
            "All messages and payments stay inside the marketplace application.",
            "Pembayaran dan komunikasi dilakukan melalui platform ini.",
            "The number in this description is a product model number, not contact information.",
        ],
    },
    "restricted_item": {
        "yes": [
            "Selling a working firearm and ammunition.",
            "Prescription medicine offered for sale without a prescription.",
            "Fake designer bag, replica brand, mirror quality 1:1, not authentic.",
            "Menjual obat resep, senjata api, atau tas bermerek palsu yang disebut replika.",
        ],
        "no": [
            "Selling an ordinary authentic secondhand item.",
            "Barang umum asli untuk kebutuhan sehari-hari, bukan senjata atau obat resep.",
            "A branded item with no evidence that it is counterfeit.",
        ],
    },
    "condition_contradiction": {
        "yes": [
            "Condition says brand new and sealed, but description says it was used for two years and is damaged.",
            "Judul menyebut baru belum dipakai, deskripsi menyebut bekas digunakan dan rusak.",
            "Title says fully working but description says it does not turn on.",
        ],
        "no": [
            "Title, description, and declared condition all agree on the item's state.",
            "Judul dan deskripsi sama-sama menyebut barang bekas yang masih berfungsi baik.",
            "Like new means previously owned with almost no signs of use.",
        ],
    },
}

CATEGORY_TEXT = {
    "electronics": "phones computers cameras audio game consoles electronic accessories",
    "fashion": "clothing shoes bags jewelry fashion accessories",
    "home": "furniture household goods kitchenware home decoration",
    "sports": "bicycles sports fitness camping outdoor equipment",
    "toys": "toys games baby items children's products",
    "books": "printed books comics magazines stationery",
    "other": "other physical goods",
}

_embedder = None
_prototypes = None
_category_vectors = None
_trained = {}
_lock = threading.RLock()


def listing_text(listing: dict) -> str:
    return (f"Title: {listing['title']}\nCategory: {listing['category']}\n"
            f"Declared condition: {listing['condition']}\nDescription: {listing['description']}")


def _vectors(texts: list[str]) -> np.ndarray:
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(ROOT / ".cache" / "models"), threads=4)
    array = np.asarray(list(_embedder.embed(texts, batch_size=32)), dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-9)


def _reference_vectors() -> tuple[dict, dict]:
    global _prototypes, _category_vectors
    if _prototypes is None:
        _prototypes = {}
        for rule, sides in PROTOTYPES.items():
            _prototypes[rule] = {side: _vectors(texts) for side, texts in sides.items()}
    if _category_vectors is None:
        names = list(CATEGORY_TEXT)
        vectors = _vectors(list(CATEGORY_TEXT.values()))
        _category_vectors = dict(zip(names, vectors))
    return _prototypes, _category_vectors


def _contrast_score(query: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    # A similarity index on 0–1 for visual comparison, NOT a calibrated probability.
    pos = float(np.max(positive @ query))
    neg = float(np.max(negative @ query))
    return round(1 / (1 + math.exp(-12 * (pos - neg))), 4)


def predict_local(listing: dict) -> dict:
    with _lock:
        prototypes, categories = _reference_vectors()
        query = _vectors([listing_text(listing)])[0]
        rules = {}
        for rule in PROTOTYPES:
            examples = prototypes[rule]
            rules[rule] = {"score": _contrast_score(query, examples["yes"], examples["no"])}
        product_query = _vectors([f"Product: {listing['title']}. {listing['description']}"])[0]
        similarities = {name: float(vector @ product_query) for name, vector in categories.items() if name != "other"}
        best = max(similarities, key=similarities.get)
        chosen = listing["category"]
        gap = max(0.0, similarities[best] - similarities.get(chosen, similarities[best]))
        rules["category_mismatch"] = {"score": round(min(1.0, gap / 0.12), 4), "suggested_category": best}
        supervised = {}
        for model_name, models in _trained.items():
            supervised[model_name] = {}
            for rule, model in models.items():
                x = [listing_text(listing)] if model_name == "tfidf" else query.reshape(1, -1)
                supervised[model_name][rule] = {"probability": round(float(model.predict_proba(x)[0, 1]), 4)}
        return {"model": MODEL_NAME, "rules": rules, "supervised": supervised, "training": training_status()}


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    with LABELS_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_label(listing: dict, labels: dict) -> int:
    if not isinstance(labels, dict) or any(labels.get(rule) not in ("yes", "no", "uncertain") for rule in RULES):
        raise ValueError("Semua empat label manual harus diisi.")
    LABELS_PATH.parent.mkdir(exist_ok=True)
    with _lock, LABELS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"listing": listing, "labels": {rule: labels[rule] for rule in RULES}}, ensure_ascii=False) + "\n")
    return len(load_labels())


def training_status() -> dict:
    records = load_labels()
    counts = {rule: {"yes": 0, "no": 0} for rule in RULES}
    for row in records:
        for rule in RULES:
            value = row["labels"].get(rule)
            if value in ("yes", "no"):
                counts[rule][value] += 1
    return {"saved_rows": len(records), "counts": counts, "trained_rules": sorted(_trained.get("tfidf", {}))}


def train_local() -> dict:
    global _trained
    with _lock:
        records = load_labels()
        embeddings = _vectors([listing_text(row["listing"]) for row in records]) if records else None
        trained = {"tfidf": {}, "bge": {}}
        for rule in RULES:
            indices = [i for i, row in enumerate(records) if row["labels"].get(rule) in ("yes", "no")]
            labels = np.asarray([int(records[i]["labels"][rule] == "yes") for i in indices])
            if len(labels) == 0 or min(np.bincount(labels, minlength=2)) < 5:
                continue
            texts = [listing_text(records[i]["listing"]) for i in indices]
            tfidf = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
            tfidf.fit(texts, labels)
            bge = LogisticRegression(max_iter=1000, class_weight="balanced")
            bge.fit(embeddings[indices], labels)
            trained["tfidf"][rule] = tfidf
            trained["bge"][rule] = bge
        _trained = trained
        return training_status()
