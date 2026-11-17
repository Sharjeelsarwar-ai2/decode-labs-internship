"""Project 4 — Image / Text Recognition (OCR)
Decode Labs AI Internship — Optional Mastery Phase.

Runs the notebook's exact pipeline (OpenCV + Tesseract) on any uploaded image,
and shows the recorded notebook run on the Vultr certificate (14 Nov 2026).
"""

import base64
import io
import shutil

import cv2
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from notebook_outputs import (
    FIG_BOXED_B64,
    FIG_COMPARISON_B64,
    RECORDED_MEAN_CONF,
    RECORDED_OPENCV,
    RECORDED_TESSERACT,
    RECORDED_TEXT,
    RECORDED_VERDICT,
    RECORDED_WORDS_ABOVE_80,
    RECORDED_WORDS_TOTAL,
)

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Project 4 — Image / Text Recognition (OCR)",
    page_icon="\U0001f50d",
    layout="wide",
)

# ---------------------------------------------------------------- styling
# Dark ink + emerald identity. Deliberately NOT the orange/cream of Project 3.
# Also hides every piece of default Streamlit chrome: header, toolbar, menu,
# footer, deploy button and the top padding they leave behind.
st.markdown(
    """
<style>
  header[data-testid="stHeader"] { visibility: hidden; height: 0; }
  div[data-testid="stToolbar"] { visibility: hidden; height: 0; }
  div[data-testid="stDecoration"] { visibility: hidden; height: 0; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  .stDeployButton { display: none; }
  .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1160px; }
  [data-testid="stAppViewContainer"] { background: #0A0F0D; }
  [data-testid="stAppViewContainer"] * { color: #E8F0EC; }
  h1, h2, h3 { color: #E8F0EC !important; letter-spacing: -0.02em; }
  h2 { border-bottom: 1px solid #1E2B25; padding-bottom: 0.4rem; }
  p, li, span, div { color: #B9C9C0; }
  .eyebrow { font-size: 0.75rem; font-weight: 700; letter-spacing: 0.22em;
             text-transform: uppercase; color: #34D399 !important; }
  .hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.05;
                color: #F2F7F4 !important; margin: 0.2rem 0 0.4rem 0; }
  .hero-sub { font-size: 1.05rem; color: #9DB4A8 !important; max-width: 720px; }
  .badge { display: inline-block; padding: 0.28rem 0.7rem; border-radius: 999px;
           border: 1px solid #24473A; background: #10201A; color: #7CE3B4 !important;
           font-size: 0.78rem; font-weight: 600; margin: 0.15rem 0.25rem 0.15rem 0; }
  .card { background: #101815; border: 1px solid #1E2B25; border-radius: 14px;
          padding: 1.1rem 1.2rem; margin-bottom: 1rem; }
  .metric-num { font-size: 2rem; font-weight: 800; color: #34D399 !important;
                line-height: 1; }
  .metric-num.warn { color: #FBBF24 !important; }
  .metric-num.bad { color: #F87171 !important; }
  .metric-label { font-size: 0.8rem; color: #8FA69B !important; margin-top: 0.3rem; }
  .verdict-pass { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
                  background: #10B981; color: #04120C !important; font-weight: 800;
                  font-size: 0.95rem; }
  .verdict-fail { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
                  background: #EF4444; color: #FFFFFF !important; font-weight: 800;
                  font-size: 0.95rem; }
  .step-num { display: inline-flex; align-items: center; justify-content: center;
              width: 1.7rem; height: 1.7rem; border-radius: 50%; background: #10B981;
              color: #04120C !important; font-weight: 800; font-size: 0.85rem;
              margin-right: 0.6rem; flex-shrink: 0; }
  .step-row { display: flex; align-items: flex-start; margin-bottom: 0.7rem; }
  .step-row b { color: #E8F0EC !important; }
  .note { font-size: 0.85rem; color: #8FA69B !important; }
  .stButton > button { background: #10B981; color: #04120C !important; border: none;
                       font-weight: 700; border-radius: 10px; padding: 0.5rem 1.2rem; }
  .stButton > button:hover { background: #34D399; color: #04120C !important; }
  .stDownloadButton > button { background: transparent; color: #34D399 !important;
                               border: 1px solid #24473A; border-radius: 10px;
                               font-weight: 600; }
  [data-testid="stFileUploader"] section { border: 1px dashed #2A4A3C !important;
                                           background: #0D1512; border-radius: 12px; }
  [data-testid="stDataFrame"] { border: 1px solid #1E2B25; border-radius: 12px; }
  code { color: #7CE3B4 !important; }
  a { color: #34D399 !important; }
  hr { border-color: #1E2B25; }
</style>
""",
    unsafe_allow_html=True,
)

MIN_CONF = 80  # the PDF's rule: 80% is the absolute minimum


# ---------------------------------------------------------------- pipeline
def run_pipeline(image_bgr):
    """The notebook's exact pipeline. Returns (stages, text, words, boxed_bgr)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)

    text = pytesseract.image_to_string(inverted, config="--psm 6")
    data = pytesseract.image_to_data(inverted, config="--psm 6",
                                     output_type=pytesseract.Output.DICT)
    words = []
    for i, word in enumerate(data["text"]):
        word = (word or "").strip()
        conf = float(data["conf"][i])
        if word and conf >= 0:
            words.append({
                "word": word,
                "conf": conf,
                "left": int(data["left"][i]),
                "top": int(data["top"][i]),
                "width": int(data["width"][i]),
                "height": int(data["height"][i]),
            })

    boxed = image_bgr.copy()
    scale = max(0.5, image_bgr.shape[1] / 900.0)
    for w in words:
        if w["conf"] >= MIN_CONF:
            x, y, bw, bh = w["left"], w["top"], w["width"], w["height"]
            cv2.rectangle(boxed, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(boxed, f"{int(w['conf'])}%", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 255, 0), 2)

    stages = {
        "Original": image_bgr,
        "Grayscale": gray,
        "Blur (5x5)": blur,
        "Threshold (Otsu)": thresh,
        "Inverted": inverted,
    }
    return stages, text, words, boxed


@st.cache_data(show_spinner=False)
def run_pipeline_cached(image_bytes):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    stages, text, words, boxed = run_pipeline(image_bgr)
    out = {}
    for name, img in stages.items():
        ok, buf = cv2.imencode(".png", img)
        out[name] = buf.tobytes() if ok else None
    ok, buf = cv2.imencode(".png", boxed)
    boxed_bytes = buf.tobytes() if ok else None
    mean_conf = float(np.mean([w["conf"] for w in words])) if words else 0.0
    above = sum(1 for w in words if w["conf"] >= MIN_CONF)
    return {
        "stages": out, "text": text, "words": words,
        "boxed": boxed_bytes, "mean_conf": mean_conf,
        "above": above, "total": len(words),
    }


def make_sample_image():
    """A clean printed-text document, generated inside the app, so the live
    demo works with no upload. Clearly a sample — the real test case (the
    Vultr certificate) is shown in the notebook section below."""
    try:
        from matplotlib import font_manager
        bold_path = font_manager.findfont("DejaVu Sans", weight="bold")
        reg_path = font_manager.findfont("DejaVu Sans")
        bold = ImageFont.truetype(bold_path, 44)
        body = ImageFont.truetype(reg_path, 26)
        small = ImageFont.truetype(reg_path, 21)
    except Exception:
        bold = ImageFont.load_default(size=44)
        body = ImageFont.load_default(size=26)
        small = ImageFont.load_default(size=21)

    img = Image.new("RGB", (1000, 640), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([28, 28, 972, 612], outline="#0F766E", width=4)
    d.text((70, 60), "SAMPLE DOCUMENT", fill="#0F172A", font=bold)
    d.text((70, 122), "Optical Character Recognition — Project 4",
           fill="#0F766E", font=body)
    d.line([70, 170, 930, 170], fill="#CBD5E1", width=2)
    lines = [
        "This sample image is generated inside the app,",
        "so you can try the pipeline without uploading",
        "anything first.",
        "",
        "Upload any image with printed text to test",
        "the same pipeline on your own document.",
        "",
        "Target: mean confidence 80% or higher.",
    ]
    y = 195
    for line in lines:
        if line:
            d.text((70, y), line, fill="#1E293B", font=body)
        y += 38
    d.text((70, 560), "Decode Labs  ·  AI Internship  ·  November 2026",
           fill="#64748B", font=small)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def b64_to_bytes(b64_string):
    return base64.b64decode(b64_string)


# ---------------------------------------------------------------- hero
st.markdown(
    """
<div class="eyebrow">Decode Labs &nbsp;·&nbsp; AI Internship &nbsp;·&nbsp; Mastery Phase</div>
<div class="hero-title">Image / Text Recognition</div>
<p class="hero-sub">Read printed text out of any image with Google's Tesseract engine —
upload a photo of a document and watch the whole pipeline run, step by step.</p>
<div>
  <span class="badge">Tesseract 5.3.4</span>
  <span class="badge">OpenCV</span>
  <span class="badge">Python</span>
  <span class="badge">PSM 6 · one uniform block of text</span>
</div>
""",
    unsafe_allow_html=True,
)

tesseract_ok = shutil.which("tesseract") is not None
if not tesseract_ok:
    st.error(
        "Tesseract is not installed on this server, so live OCR cannot run here. "
        "On Streamlit Cloud it comes from `packages.txt` (`tesseract-ocr`). "
        "The recorded notebook run below still works."
    )

st.markdown("---")

# ---------------------------------------------------------------- live demo
st.markdown("## Try it live")
st.markdown(
    "Upload an image with printed text, or run the built-in sample. "
    "The pipeline is the notebook's exact code: grayscale, Gaussian blur, "
    "Otsu threshold, invert, then Tesseract."
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
with col_sample:
    use_sample = st.button("Use built-in sample", use_container_width=True)

image_bytes = None
image_source = ""
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    image_source = uploaded.name
elif use_sample or "sample_bytes" not in st.session_state:
    st.session_state["sample_bytes"] = make_sample_image()
    image_bytes = st.session_state["sample_bytes"]
    image_source = "Built-in sample document"
elif "sample_bytes" in st.session_state:
    image_bytes = st.session_state["sample_bytes"]
    image_source = "Built-in sample document"

if image_bytes:
    if not tesseract_ok:
        st.warning("Live OCR needs the Tesseract binary — see the note above.")
    else:
        with st.spinner("Reading text…"):
            result = run_pipeline_cached(image_bytes)

        mean_conf = result["mean_conf"]
        passed = mean_conf >= MIN_CONF
        verdict_cls = "verdict-pass" if passed else "verdict-fail"
        verdict_txt = "PASS" if passed else "FAIL"
        num_cls = "" if passed else "bad"

        st.markdown(
            f"""
<div class="card">
  <div style="display:flex; flex-wrap:wrap; gap: 2rem; align-items: center;">
    <div><div class="metric-num {num_cls}">{mean_conf:.1f}%</div>
         <div class="metric-label">Mean confidence</div></div>
    <div><div class="metric-num">{result["above"]} / {result["total"]}</div>
         <div class="metric-label">Words at 80% or higher</div></div>
    <div><div class="metric-num">{result["total"]}</div>
         <div class="metric-label">Words found</div></div>
    <div><span class="{verdict_cls}">{verdict_txt}</span>
         <div class="metric-label">80% minimum required</div></div>
  </div>
  <p class="note" style="margin-top: 0.8rem;">Source: {image_source}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### Pipeline stages")
        stage_cols = st.columns(5)
        for col, name in zip(stage_cols,
                             ["Original", "Grayscale", "Blur (5x5)",
                              "Threshold (Otsu)", "Inverted"]):
            with col:
                st.image(result["stages"][name], caption=name,
                         use_container_width=True)

        st.markdown("### What the machine read")
        text_col, words_col = st.columns([1, 1])
        with text_col:
            st.code(result["text"].strip() or "(no text detected)",
                    language=None)
            st.download_button(
                "Download extracted text (.txt)",
                data=result["text"],
                file_name="extracted-text.txt",
                mime="text/plain",
            )
        with words_col:
            if result["words"]:
                import pandas as pd

                df = pd.DataFrame([
                    {"Word": w["word"],
                     "Confidence %": round(w["conf"], 1),
                     "Status": "✓ kept" if w["conf"] >= MIN_CONF else "✗ dropped"}
                    for w in result["words"]
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("No words detected in this image.")

        st.markdown("### Detected words — green boxes, 80% and up")
        st.image(result["boxed"],
                 caption="Green box = confidence 80% or higher, with the score on top.",
                 use_container_width=True)
        st.download_button(
            "Download boxed image (.png)",
            data=result["boxed"],
            file_name="detected-words.png",
            mime="image/png",
        )

st.markdown("---")

# ---------------------------------------------------------------- notebook run
st.markdown("## The notebook run — Vultr certificate, 14 Nov 2026")
st.markdown(
    "Project 4's real test case was my own **Vultr Agent Rush Certificate of "
    "Completion** — deliberately hard: light text on a dark blue gradient. "
    "This is the notebook's own recorded output from its Google Colab run, "
    "figures and numbers exactly as they came out."
)

st.markdown(
    f"""
<div class="card">
  <div style="display:flex; flex-wrap:wrap; gap: 2rem; align-items: center;">
    <div><div class="metric-num">{RECORDED_MEAN_CONF}%</div>
         <div class="metric-label">Mean confidence (recorded)</div></div>
    <div><div class="metric-num">{RECORDED_WORDS_ABOVE_80} / {RECORDED_WORDS_TOTAL}</div>
         <div class="metric-label">Words at 80% or higher</div></div>
    <div><span class="verdict-pass">{RECORDED_VERDICT}</span>
         <div class="metric-label">80% minimum required</div></div>
    <div><div class="metric-label">OpenCV {RECORDED_OPENCV} · Tesseract {RECORDED_TESSERACT}</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Original vs pre-processed — the notebook's comparison")
st.image(b64_to_bytes(FIG_COMPARISON_B64),
         caption="Left: the Vultr certificate as uploaded. Right: the same image after "
                 "grayscale, blur and Otsu threshold — black and white, ready for Tesseract.",
         use_container_width=True)

st.markdown("### Detected words on the certificate")
st.image(b64_to_bytes(FIG_BOXED_B64),
         caption="The notebook's boxed detection figure — green boxes around every word "
                 "Tesseract read at 80% confidence or higher.",
         use_container_width=True)

st.markdown("### Text the notebook extracted")
st.code(RECORDED_TEXT, language=None)

with st.expander("Honest limitations — what didn't work"):
    st.markdown(
        """
- The big **"CERTIFICATE OF COMPLETION"** header was not detected — PSM 6
          assumes one uniform block of text and sometimes skips a large title.
          Trying `--psm 3` (fully automatic) is the suggested fix.
- The small glitches (a stray `.`, `F` where ANTIGRAVITY split across lines,
          `HACKATHON i`) are the 3 words that fell below 80% — all from the small
          body text and the gradient background.
- The bottom of the certificate (date / signature area) is lost in the
          thresholded image: one global Otsu threshold can't separate text from
          a gradient background everywhere. A normal page (dark text on white
          paper) is the easy case; a dark certificate with a gradient is the hard case.
        """
    )

st.markdown("---")

# ---------------------------------------------------------------- how it works
st.markdown("## How it works")
st.markdown(
    """
<div class="step-row"><span class="step-num">1</span><span><b>Load</b> the image with OpenCV.</span></div>
<div class="step-row"><span class="step-num">2</span><span><b>Grayscale</b> — colour carries no information for reading text, so 3 channels collapse into 1.</span></div>
<div class="step-row"><span class="step-num">3</span><span><b>Gaussian blur (5×5)</b> — smooths out tiny specks so the next step doesn't mistake a speck for a letter.</span></div>
<div class="step-row"><span class="step-num">4</span><span><b>Otsu threshold</b> — forces every pixel to pure black or white; Otsu finds the cutoff automatically.</span></div>
<div class="step-row"><span class="step-num">5</span><span><b>Invert</b> — the certificate has light text on a dark background; Tesseract expects dark text on light, so flip it.</span></div>
<div class="step-row"><span class="step-num">6</span><span><b>OCR</b> with Tesseract, <code>--psm 6</code> (one uniform block of text).</span></div>
<div class="step-row"><span class="step-num">7</span><span><b>Confidence check</b> — every word gets a score; the rule is 80% minimum, and words below it are dropped.</span></div>
<div class="step-row"><span class="step-num">8</span><span><b>Visual confirmation</b> — a green box around every detected word (80%+), with its confidence written on top.</span></div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")
st.markdown(
    """
<p class="note">Built by Muhammad Sharjeel Sarwar — Decode Labs AI Internship, Project 4 (Mastery Phase), November 2026.
&nbsp;·&nbsp; <a href="https://github.com/Sharjeelsarwar-ai2/decode-labs-internship/blob/main/Project-4-Image-Text-Recognition/Decodelabs_project_4.ipynb">Notebook on GitHub</a>
&nbsp;·&nbsp; <a href="https://github.com/Sharjeelsarwar-ai2/decode-labs-internship/tree/main/Project-4-Image-Text-Recognition">Project folder</a></p>
""",
    unsafe_allow_html=True,
)
