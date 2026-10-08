"""Local, text-only marketplace moderation sandbox using OpenRouter Jev."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import requests
import local_moderation

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "moderation_ui"
ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
DECIDER_MODEL = "perplexity/pplx-decider-v1.1-27b"
OPENAI_DECISIONS_MODEL = "openai/gpt-6-luna-decisions"
CLEF_MODEL = "cloudflare/clef"
DECISION_MODELS = {
    "jev": MODEL,
    "decider": DECIDER_MODEL,
    "openai": OPENAI_DECISIONS_MODEL,
    "clef": CLEF_MODEL,
}
RUNTIME_KEY = ""
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "") == "1"
PUBLIC_DAILY_REQUESTS_PER_VISITOR = 10
_public_usage = {}
_public_usage_lock = threading.Lock()
_public_inference_slots = threading.BoundedSemaphore(2)


def allow_public_request(client_address: str) -> bool:
    """Best-effort CPU protection; OpenRouter's daily guardrail is the spend cap."""
    day = datetime.now(timezone.utc).date().isoformat()
    visitor = hashlib.sha256(client_address.encode("utf-8")).hexdigest()
    with _public_usage_lock:
        if _public_usage and next(iter(_public_usage))[0] != day:
            _public_usage.clear()
        if len(_public_usage) > 10000:
            _public_usage.clear()
        count = _public_usage.get((day, visitor), 0)
        if count >= PUBLIC_DAILY_REQUESTS_PER_VISITOR:
            return False
        _public_usage[(day, visitor)] = count + 1
        return True

CATEGORIES = {
    "electronics": "Ponsel, komputer, kamera, audio, konsol game, dan aksesorinya.",
    "fashion": "Pakaian, sepatu, tas, perhiasan, dan aksesori fesyen.",
    "home": "Perabot, perlengkapan rumah, dekorasi, dan peralatan dapur.",
    "sports": "Sepeda, alat olahraga, perlengkapan kemah, dan perlengkapan aktivitas luar ruang.",
    "toys": "Mainan, permainan, perlengkapan bayi, dan perlengkapan anak.",
    "books": "Buku cetak, majalah, komik, dan alat tulis.",
    "other": "Barang fisik umum yang tidak termasuk kategori lain di daftar ini.",
}
CONDITIONS = {
    "new": "Baru: belum pernah dipakai.",
    "like_new": "Seperti baru: pernah dimiliki atau dipakai tetapi hampir tanpa bekas.",
    "good": "Baik: berfungsi, mungkin ada bekas pemakaian ringan.",
    "fair": "Cukup: berfungsi tetapi ada kerusakan atau bekas pemakaian yang jelas.",
    "for_parts": "Untuk suku cadang/perbaikan: tidak berfungsi atau tidak lengkap.",
}
RESTRICTED = {
    "none": "Barang tidak termasuk tiga jenis larangan studi di bawah ini. Penyebutan nama merek saja tidak membuktikan barang palsu.",
    "weapon": "Senjata api, amunisi, bagian senjata api, atau barang yang jelas dipasarkan sebagai senjata untuk melukai orang. Mainan, replika dekoratif yang tidak berfungsi, dan alat rumah tangga biasa tidak otomatis masuk.",
    "prescription_drug": "Obat resep atau zat terkontrol yang ditawarkan untuk dijual. Suplemen biasa dan perlengkapan kesehatan umum tidak otomatis masuk.",
    "explicit_counterfeit": "Barang yang penjual sendiri nyatakan tiruan, palsu, replika merek, mirror quality, 1:1, AAA, atau same factory sebagai pengganti barang bermerek asli. Harga murah atau nama merek saja tidak cukup.",
}
RULE_LABELS = {
    "off_platform": "Transaksi di luar platform",
    "restricted_item": "Barang terlarang menurut kebijakan demo",
    "category_mismatch": "Kategori tidak sesuai",
    "condition_contradiction": "Informasi listing bertentangan",
}


def validate_listing(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Format listing harus berupa objek JSON.")
    title = data.get("title", "")
    description = data.get("description", "")
    category = data.get("category", "")
    condition = data.get("condition", "")
    if not isinstance(title, str) or not 3 <= len(title.strip()) <= 160:
        raise ValueError("Judul harus berisi 3–160 karakter.")
    maximum_description = 1000 if PUBLIC_DEMO else 5000
    if not isinstance(description, str) or not 10 <= len(description.strip()) <= maximum_description:
        raise ValueError(f"Deskripsi harus berisi 10–{maximum_description:,} karakter.")
    if not isinstance(category, str) or not isinstance(condition, str) or category not in CATEGORIES or condition not in CONDITIONS:
        raise ValueError("Kategori atau kondisi tidak dikenal.")
    return {
        "title": title.strip(),
        "description": description.strip(),
        "category": category,
        "condition": condition,
    }


def make_payload(listing: dict, model: str = MODEL) -> dict:
    return {
        "model": model,
        "state": {
            "listing": {
                "title": listing["title"],
                "description": listing["description"],
                "category": listing["category"],
                "declared_condition": listing["condition"],
            },
            "category_definition": CATEGORIES[listing["category"]],
            "condition_definition": CONDITIONS[listing["condition"]],
            "study_policy": "Kebijakan demo terbatas pada tiga jenis larangan yang dijelaskan dalam pilihan restricted_item. Jangan menyimpulkan keaslian barang dari harga, nama merek, atau asumsi eksternal.",
        },
        "questions": {
            "off_platform": {
                "type": "noul",
                "instructions": "Apakah `listing.title` atau `listing.description` mengajak pembeli menghubungi penjual, membayar, atau menyelesaikan transaksi di luar marketplace? Nilai maksud transaksi, bukan sekadar keberadaan angka atau tautan.",
                "criteria": {
                    "true": "Ada ajakan bertransaksi atau berkomunikasi di luar platform, termasuk nomor kontak, akun pesan, tautan eksternal untuk transaksi, atau permintaan pembayaran langsung.",
                    "false": "Tidak ada ajakan di luar platform. Nomor model produk, tautan manual/spesifikasi, atau pengambilan barang yang diatur melalui platform tetap negatif.",
                },
            },
            "restricted_item": {
                "type": "choice",
                "instructions": "Menurut `study_policy`, jenis larangan mana yang secara eksplisit didukung oleh barang di `listing.title` dan `listing.description`? Pilih none jika tidak ada bukti yang cukup. Jangan menganggap merek atau harga sebagai bukti barang palsu.",
                "criteria": RESTRICTED,
            },
            "category_mismatch": {
                "type": "noul",
                "instructions": "Apakah barang dalam `listing.title` dan `listing.description` jelas tidak sesuai dengan `category_definition` untuk kategori yang dipilih?",
                "criteria": {
                    "true": "Jenis barang jelas berada di luar definisi kategori yang dipilih.",
                    "false": "Kategori yang dipilih masuk akal, walaupun ada kategori lain yang mungkin lebih spesifik, atau informasi terlalu sedikit untuk memastikan ketidaksesuaian.",
                },
            },
            "condition_contradiction": {
                "type": "noul",
                "instructions": "Apakah ada pertentangan eksplisit antara `listing.title`, `listing.description`, dan kondisi yang dijelaskan oleh `condition_definition`? Periksa klaim baru/bekas/kerusakan serta fakta barang lain yang jelas berbeda.",
                "criteria": {
                    "true": "Teks saling bertentangan secara jelas, misalnya disebut baru tetapi ternyata pernah dipakai, atau judul menyebut berfungsi sedangkan deskripsi menyebut rusak.",
                    "false": "Deskripsi hanya menambahkan rincian tanpa bertentangan; ungkapan seperti 'seperti baru' dapat konsisten dengan barang bekas berkondisi baik.",
                },
            },
        },
    }


def parse_answers(body: dict) -> dict:
    answers = body.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Respons Jev tidak memiliki answers yang valid.")
    output = {}
    for name in ("off_platform", "category_mismatch", "condition_contradiction"):
        answer = answers.get(name)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise ValueError(f"Respons Jev untuk {name} tidak valid.")
        score = float(answer.get("noul", -1))
        if not 0 <= score <= 1:
            raise ValueError(f"Probabilitas {name} di luar rentang.")
        output[name] = {"probability": score}
    restricted = answers.get("restricted_item")
    if not isinstance(restricted, dict) or restricted.get("type") != "choice":
        raise ValueError("Respons Jev untuk restricted_item tidak valid.")
    choice = restricted.get("choice")
    raw = restricted.get("probabilities")
    if choice not in RESTRICTED or not isinstance(raw, dict):
        raise ValueError("Pilihan barang terlarang tidak valid.")
    probabilities = {key: float(raw.get(key, 0)) for key in RESTRICTED}
    if any(not 0 <= p <= 1 for p in probabilities.values()) or not 0.975 <= sum(probabilities.values()) <= 1.025:
        raise ValueError("Distribusi barang terlarang tidak valid.")
    output["restricted_item"] = {
        "choice": choice,
        "probability": 1 - probabilities["none"],
        "probabilities": probabilities,
    }
    return output


def route(rules: dict) -> tuple[str, list[str]]:
    """Provisional demo routing; thresholds must be validated before any use."""
    flagged = [RULE_LABELS[name] for name, item in rules.items() if item["probability"] >= 0.65]
    if rules["restricted_item"]["probability"] >= 0.98:
        return "reject", flagged
    if flagged or any(0.35 <= item["probability"] < 0.65 for item in rules.values()):
        return "hold", flagged
    return "publish", flagged


def evaluate_listing(listing: dict, key: str, model: str = MODEL) -> dict:
    started = time.perf_counter()
    try:
        response = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=make_payload(listing, model),
            timeout=45,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Koneksi ke OpenRouter gagal atau waktu tunggu habis.") from exc
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter mengembalikan HTTP {response.status_code}.")
    try:
        body = response.json()
        rules = parse_answers(body)
    except (ValueError, TypeError, KeyError) as exc:
        raise RuntimeError("Format respons Jev tidak sesuai dengan empat aturan.") from exc
    action, flagged = route(rules)
    usage = body.get("usage") or {}
    return {
        "action": action,
        "flagged": flagged,
        "rules": rules,
        "model": body.get("model", model),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "cost_usd": usage.get("cost"),
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "policy_note": "Simulasi dengan ambang sementara; belum dikalibrasi pada data moderasi berlabel.",
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if path == "/api/status":
            self.send_json(200, {"key_ready": bool(current_key()), "model": MODEL, "training": local_moderation.training_status(), "public_demo": PUBLIC_DEMO})
            return
        if path not in files:
            self.send_error(404)
            return
        name, content_type = files[path]
        raw = (STATIC / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; base-uri 'none'; form-action 'self'")
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        if self.path not in ("/api/evaluate", "/api/local", "/api/key", "/api/label", "/api/train"):
            self.send_error(404)
            return
        if PUBLIC_DEMO and self.path in ("/api/key", "/api/label", "/api/train"):
            self.send_error(404)
            return
        origin = self.headers.get("Origin")
        host = self.headers.get("Host", "")
        if origin and origin not in (f"http://{host}", f"https://{host}"):
            self.send_json(403, {"error": "Origin tidak diizinkan."})
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.send_json(415, {"error": "Gunakan application/json."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16000:
                raise ValueError("Ukuran permintaan tidak valid.")
            data = json.loads(self.rfile.read(length))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        if self.path == "/api/key":
            global RUNTIME_KEY
            value = data.get("key") if isinstance(data, dict) else None
            if not isinstance(value, str) or not 10 <= len(value.strip()) <= 500:
                self.send_json(400, {"error": "Masukkan API key OpenRouter yang valid."})
                return
            RUNTIME_KEY = value.strip()
            self.send_json(200, {"key_ready": True})
            return
        if self.path == "/api/train":
            try:
                self.send_json(200, {"training": local_moderation.train_local()})
            except Exception:
                self.send_json(500, {"error": "Pelatihan model lokal gagal. Periksa data label dan terminal."})
            return
        try:
            listing = validate_listing(data.get("listing") if self.path == "/api/label" and isinstance(data, dict) else data)
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        if self.path == "/api/label":
            try:
                count = local_moderation.save_label(listing, data.get("labels"))
                self.send_json(200, {"saved_rows": count, "training": local_moderation.training_status()})
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if PUBLIC_DEMO:
            # Caddy supplies the last address; the Python server listens only on loopback.
            forwarded = self.headers.get("X-Forwarded-For", "")
            visitor = forwarded.split(",")[-1].strip() if forwarded else self.client_address[0]
            if not _public_inference_slots.acquire(blocking=False):
                self.send_json(503, {"error": "Server sedang sibuk. Coba lagi sebentar."})
                return
            if not allow_public_request(visitor):
                _public_inference_slots.release()
                self.send_json(429, {"error": "Batas 10 pemeriksaan per pengunjung per hari telah tercapai."})
                return
        try:
            local = local_moderation.predict_local(listing)
        except Exception:
            self.send_json(500, {"error": "Model lokal gagal dijalankan. Periksa terminal dan cache model."})
            return
        finally:
            if PUBLIC_DEMO:
                _public_inference_slots.release()
        if self.path == "/api/local":
            self.send_json(200, {"local": local})
            return
        requested = data.get("decision_models", list(DECISION_MODELS)) if isinstance(data, dict) else []
        if not isinstance(requested, list) or not requested or any(not isinstance(name, str) or name not in DECISION_MODELS for name in requested) or len(set(requested)) != len(requested):
            self.send_json(400, {"error": "Pilih minimal satu model keputusan yang tersedia."})
            return
        key = current_key()
        if not key:
            result = {"local": local}
            for name in requested:
                result[name] = None
                result[name + "_error"] = "API key belum disetel; hasil lokal tetap tersedia."
            self.send_json(200, result)
            return
        result = {"local": local}
        with ThreadPoolExecutor(max_workers=len(requested)) as pool:
            futures = {name: pool.submit(evaluate_listing, listing, key, DECISION_MODELS[name]) for name in requested}
            for name, future in futures.items():
                try:
                    result[name] = future.result()
                    result[name + "_error"] = None
                except RuntimeError as exc:
                    result[name] = None
                    result[name + "_error"] = str(exc)
        self.send_json(200, result)

    def log_message(self, format: str, *args: object) -> None:
        # Do not log listing contents or Authorization headers.
        print("[moderation] " + format % args)


def current_key() -> str:
    return ("" if PUBLIC_DEMO else RUNTIME_KEY) or os.getenv("OPENROUTER_API_KEY", "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Local Jev listing moderation sandbox")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if PUBLIC_DEMO:
        if not os.getenv("OPENROUTER_API_KEY", "").strip():
            parser.error("PUBLIC_DEMO requires OPENROUTER_API_KEY")
        if not local_moderation.LABELS_PATH.is_file():
            parser.error("PUBLIC_DEMO requires MODERATION_LABELS_PATH with teacher labels")
        status = local_moderation.train_local()
        if len(status["trained_rules"]) != len(local_moderation.RULES):
            parser.error("PUBLIC_DEMO requires trained local models for all four rules")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Buka http://127.0.0.1:{args.port} di browser. Ctrl+C untuk berhenti.", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
