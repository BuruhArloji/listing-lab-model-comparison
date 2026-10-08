"""Public Listing Lab frontend for Streamlit Community Cloud."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from html import escape
import os
from pathlib import Path

import streamlit as st

import local_moderation
from moderation_app import DECISION_MODELS, evaluate_listing, validate_listing

ROOT = Path(__file__).resolve().parent
PUBLIC_LABELS = ROOT / "public_data" / "teacher_labels.jsonl"
MODEL_LABELS = {
    "jev": "Jev 1.13",
    "decider": "Perplexity Decider V1.1",
    "openai": "OpenAI GPT-6 Luna",
    "clef": "Cloudflare Clef",
}
RULE_LABELS = {
    "off_platform": "Transaksi di luar platform",
    "restricted_item": "Barang terlarang",
    "category_mismatch": "Kategori tidak sesuai",
    "condition_contradiction": "Informasi bertentangan",
}
CATEGORY_LABELS = {
    "electronics": "Elektronik", "fashion": "Fashion", "home": "Rumah & dapur",
    "sports": "Olahraga & outdoor", "toys": "Mainan & anak",
    "books": "Buku & alat tulis", "other": "Lainnya",
}
CONDITION_LABELS = {
    "new": "Baru", "like_new": "Seperti baru", "good": "Baik",
    "fair": "Cukup", "for_parts": "Untuk suku cadang",
}
EXAMPLES = {
    "Normal": ("Sepatu lari pria ukuran 42", "fashion", "good", "Sepatu lari warna biru ukuran 42. Sudah dipakai beberapa kali, sol masih bagus dan tidak ada bagian yang rusak. Pembayaran dan pengiriman melalui marketplace ini."),
    "Kontak luar": ("Kamera mirrorless dengan lensa kit", "electronics", "good", "Kamera berfungsi normal dan lengkap dengan charger. Untuk harga lebih murah, chat saya lewat WhatsApp 0812-3456-7890 lalu bayar langsung di luar aplikasi."),
    "Kontradiksi": ("Laptop baru segel, belum pernah dipakai", "electronics", "new", "Laptop ini sudah saya pakai selama dua tahun. Baterai melemah dan ada retak di sudut casing, tetapi masih menyala dan bisa dipakai kerja."),
    "Barang terlarang": ("Tas desainer mirror quality 1:1", "fashion", "new", "Tas replika merek terkenal, kualitas mirror 1:1 dan bukan produksi resmi merek aslinya. Kondisi baru, belum pernah digunakan."),
}


def apply_example(name: str) -> None:
    title, category, condition, description = EXAMPLES[name]
    st.session_state.update(title=title, category=category, condition=condition, description=description)
    st.session_state.pop("comparison", None)
    st.session_state.pop("comparison_error", None)


def secret_value(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except FileNotFoundError:
        value = ""
    return str(value or os.getenv(name, "")).strip()


def openrouter_key() -> str:
    if secret_value("OPENROUTER_DAILY_GUARDRAIL_CONFIRMED").lower() != "true":
        return ""
    return secret_value("OPENROUTER_API_KEY")


@st.cache_resource(show_spinner="Menyiapkan model lokal dari 926 label sintetis…")
def trained_models() -> dict:
    if not PUBLIC_LABELS.is_file():
        raise FileNotFoundError("public_data/teacher_labels.jsonl tidak ditemukan")
    local_moderation.LABELS_PATH = PUBLIC_LABELS
    status = local_moderation.train_local()
    if len(status["trained_rules"]) != 4:
        raise RuntimeError("Data publik belum cukup untuk melatih keempat aturan")
    return {"models": local_moderation._trained, "status": status}


def run_comparison(listing: dict, selected: list[str]) -> dict:
    resources = trained_models()
    local_moderation._trained = resources["models"]
    result = {"local": local_moderation.predict_local(listing)}
    key = openrouter_key()
    if not key:
        for name in selected:
            result[name] = None
            result[name + "_error"] = "Key atau konfirmasi guardrail US$0,01/hari belum disetel di Secrets."
        return result
    if selected:
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures = {name: pool.submit(evaluate_listing, listing, key, DECISION_MODELS[name]) for name in selected}
            for name, future in futures.items():
                try:
                    result[name] = future.result()
                    result[name + "_error"] = None
                except RuntimeError as exc:
                    result[name] = None
                    result[name + "_error"] = str(exc)
    return result


def score_html(value: float | None) -> str:
    if value is None:
        return '<span class="muted">—</span>'
    color = "bad" if value >= .65 else "medium" if value >= .35 else "good"
    percentage = round(value * 100)
    return f'<strong class="{color}">{percentage}%</strong><span class="meter"><i class="{color}" style="width:{percentage}%"></i></span>'


def build_matrix(result: dict) -> str:
    local = result["local"]
    columns = [
        ("BGE lokal", "Kemiripan tanpa label", lambda rule: local["rules"].get(rule, {}).get("score"), "Tanpa aksi", "Lokal", "—"),
    ]
    for model, title in (("tfidf", "TF-IDF + LogReg"), ("bge", "BGE + LogReg")):
        rules = local.get("supervised", {}).get(model, {})
        count = len(rules)
        columns.append((title, "ML lokal terlatih", lambda rule, rules=rules: rules.get(rule, {}).get("probability"), f"{count}/4 aturan", "Lokal", "—"))
    for name, title in MODEL_LABELS.items():
        output = result.get(name)
        rules = output.get("rules", {}) if isinstance(output, dict) else {}
        status = {"publish": "Lolos", "hold": "Tinjau", "reject": "Tolak"}.get(output.get("action"), "Periksa") if output else result.get(name + "_error", "Tidak dipilih")
        elapsed = f'{output["latency_ms"]:,} ms' if output and output.get("latency_ms") is not None else "—"
        cost = f'${float(output["cost_usd"]):.6f}' if output and output.get("cost_usd") is not None else "—"
        columns.append((title, "OpenRouter", lambda rule, rules=rules: rules.get(rule, {}).get("probability"), status, elapsed, cost))

    parts = ['<div class="matrix-wrap"><table class="matrix"><thead><tr><th>Aspek</th>']
    for title, kind, *_ in columns:
        parts.append(f'<th>{escape(title)}<small>{escape(kind)}</small></th>')
    parts.append('</tr></thead><tbody>')
    parts.append('<tr><th>Keputusan</th>')
    for _, _, _, status, _, _ in columns:
        parts.append(f'<td class="status">{escape(str(status))}</td>')
    parts.append('</tr>')
    for rule, label in RULE_LABELS.items():
        parts.append(f'<tr><th>{escape(label)}</th>')
        for _, _, getter, _, _, _ in columns:
            value = getter(rule)
            parts.append(f'<td>{score_html(float(value) if value is not None else None)}</td>')
        parts.append('</tr>')
    for label, index in (("Waktu respons", 4), ("Biaya API", 5)):
        parts.append(f'<tr><th>{label}</th>')
        for column in columns:
            parts.append(f'<td>{escape(str(column[index]))}</td>')
        parts.append('</tr>')
    parts.append('<tr><th>Estimasi request / US$1</th>')
    for column in columns:
        cost = column[5]
        estimate = f'≈ {int(1 / float(cost[1:])):,}' if cost.startswith("$") and float(cost[1:]) > 0 else "—"
        parts.append(f'<td>{escape(estimate)}</td>')
    parts.append('</tr></tbody></table></div>')
    return "".join(parts)


st.set_page_config(page_title="Listing Lab · Perbandingan Model", page_icon="◇", layout="wide")
st.markdown("""
<style>
.stApp{background:#101718;color:#e8ede9}
.block-container{max-width:1400px;padding-top:2.5rem;padding-bottom:3rem}
h1,h2,h3{letter-spacing:-.035em}
.eyebrow{color:#bde686;font-size:.75rem;font-weight:700;letter-spacing:.17em}
.intro{color:#a7b6b0;max-width:760px;line-height:1.6}
.budget{border-left:3px solid #a8d679;background:#1e2d27;padding:12px 16px;border-radius:6px;color:#cbdcc6;margin:1rem 0 1.8rem}
[data-testid="stVerticalBlockBorderWrapper"]{width:100%}
.matrix-wrap{width:100%;overflow-x:auto;border:1px solid #3d5148;border-radius:10px;margin-top:12px}
.matrix{border-collapse:collapse;min-width:1320px;width:100%;font-size:.82rem;color:#deeadf}
.matrix th,.matrix td{border-right:1px solid #34463d;border-bottom:1px solid #34463d;padding:12px 13px;text-align:left;min-width:158px;vertical-align:top}
.matrix th:first-child{min-width:180px;position:sticky;left:0;background:#20342c;z-index:1}
.matrix thead th{background:#263e31;color:#e7f5e6}
.matrix thead th:first-child{z-index:2}
.matrix small{display:block;color:#9eb4a2;font-size:.7rem;font-weight:400;margin-top:4px}
.matrix .status{font-weight:600;color:#d5e5c9;word-break:break-word}
.matrix strong{display:block}.matrix .good{color:#bde88f}.matrix .medium{color:#f1d28e}.matrix .bad{color:#efab98}
.meter{display:block;background:#395148;height:4px;border-radius:9px;margin-top:8px}.meter i{display:block;height:100%;border-radius:9px;background:currentColor}
.muted{color:#869b8e}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">LABORATORIUM MODERASI LISTING</div>', unsafe_allow_html=True)
st.title("Tulis listing Anda. Bandingkan keputusannya.")
st.markdown('<p class="intro">Bandingkan Jev, Perplexity Decider, OpenAI, Clef, BGE lokal, serta model TF-IDF dan BGE yang dilatih dari label sintetis pada empat aturan moderasi yang sama.</p>', unsafe_allow_html=True)
st.markdown('<div class="budget">Demo publik · Listing tidak disimpan · Target anggaran OpenRouter bersama US$0,01 per hari. Model cloud aktif setelah guardrail key dikonfirmasi; model lokal selalu bisa dicoba.</div>', unsafe_allow_html=True)

for field, default in (("title", ""), ("category", "electronics"), ("condition", "new"), ("description", "")):
    st.session_state.setdefault(field, default)

with st.container(border=True):
    st.caption("01 / INPUT")
    st.subheader("Buat listing")
    st.caption("Coba contoh:")
    example_cols = st.columns(4)
    for col, name in zip(example_cols, EXAMPLES):
        with col:
            st.button(name, key=f"example-{name}", on_click=apply_example, args=(name,), use_container_width=True)
    with st.form("listing-form"):
        st.text_input("Judul produk", key="title", max_chars=160)
        category_col, condition_col = st.columns(2)
        with category_col:
            st.selectbox("Kategori", list(CATEGORY_LABELS), format_func=lambda key: CATEGORY_LABELS[key], key="category")
        with condition_col:
            st.selectbox("Kondisi", list(CONDITION_LABELS), format_func=lambda key: CONDITION_LABELS[key], key="condition")
        st.text_area("Deskripsi", key="description", max_chars=1000, height=160)
        selected = st.multiselect("Model keputusan OpenRouter", list(MODEL_LABELS), default=list(MODEL_LABELS), format_func=lambda key: MODEL_LABELS[key])
        st.caption("Setiap model yang dipilih membuat satu request API. BGE lokal selalu berjalan. Kosongkan pilihan untuk mencoba model lokal saja.")
        submitted = st.form_submit_button("Uji listing ↗", type="primary")

if submitted:
    try:
        listing = validate_listing({field: st.session_state[field] for field in ("title", "category", "condition", "description")})
        with st.spinner("Model sedang membaca listing…"):
            st.session_state["comparison"] = run_comparison(listing, selected)
        st.session_state.pop("comparison_error", None)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        st.session_state.pop("comparison", None)
        st.session_state["comparison_error"] = str(exc)

with st.container(border=True):
    st.caption("02 / ANALISIS")
    st.subheader("Hasil pemeriksaan")
    if error := st.session_state.get("comparison_error"):
        st.error(error)
    if result := st.session_state.get("comparison"):
        st.markdown("**Perbandingan sejajar** · geser tabel ke samping untuk melihat semua model.")
        st.markdown(build_matrix(result), unsafe_allow_html=True)
        st.caption("BGE tanpa label menunjukkan kemiripan, bukan probabilitas terkalibrasi. Model terlatih meniru pseudo-label Decider; ini belum mengukur accuracy terhadap label manusia. Biaya dan estimasi request berubah mengikuti panjang teks dan harga model.")
    else:
        st.info("Isi formulir lalu tekan **Uji listing**. Hasil setiap model akan muncul di sini.")
