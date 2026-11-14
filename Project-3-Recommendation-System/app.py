"""Tech Stack Recommender — Decode Labs Project 3 Capstone.

Content-based recommendation: the user picks exactly 3 skills they have,
each job role is a document made of its skills, TF-IDF weights the skills
(rare skills count more), cosine similarity ranks the roles, and the app
returns the Top 3 career paths with matched skills and skills to learn.
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------- page setup
st.set_page_config(
    page_title="Tech Stack Recommender",
    page_icon="🧭",
    layout="wide",
)

# ------------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
/* hide only the toolbar's right-side buttons (Share / Deploy) — NOT the
   toolbar itself: the sidebar's expand button renders inside it */
[data-testid="stToolbarActions"] { display: none !important; }
[data-testid="stAppDeployButton"] { display: none !important; }
.block-container { padding-top: 1.2rem; max-width: 1080px; }

/* warm paper page wash */
.stApp { background: linear-gradient(180deg, #FBF1DF 0%, #FAF6EF 220px, #FAF6EF 100%); }

/* hero banner — warm ember gradient, off-centre glow + film grain
   (grain + serif type are what keep it from reading as AI-generated) */
.hero {
    position: relative;
    background:
      url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='180' height='180' filter='url(%23n)' opacity='0.16'/></svg>"),
      radial-gradient(ellipse 55% 85% at 88% -10%, rgba(255, 196, 110, 0.55), rgba(255, 196, 110, 0) 65%),
      radial-gradient(ellipse 70% 90% at 0% 110%, rgba(120, 40, 8, 0.5), rgba(120, 40, 8, 0) 60%),
      linear-gradient(118deg, #2E1A0E 0%, #7A3416 46%, #C05A1E 78%, #DE8A2E 100%);
    border-radius: 22px;
    padding: 34px 36px 30px 36px;
    color: #FFF8EC;
    box-shadow: 0 18px 40px rgba(122, 52, 22, 0.30);
    margin-bottom: 26px;
}
.hero h1 { color: #FFF8EC !important; font-family: 'Fraunces', Georgia, serif !important;
           font-size: 2.5rem !important; font-weight: 700;
           letter-spacing: -0.01em; margin: 0 0 6px 0; line-height: 1.08; }
.hero p { color: rgba(255, 244, 224, 0.92); font-size: 1.05rem; margin: 0 0 18px 0; max-width: 640px; }
.hero .stat { display: inline-block; background: rgba(255, 248, 236, 0.14);
              border: 1px solid rgba(255, 248, 236, 0.32); color: #FFF8EC;
              border-radius: 999px; padding: 6px 14px; font-size: 0.83rem;
              font-weight: 600; margin-right: 8px; }

/* section heading */
.section-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; font-weight: 700; color: #2E1A0E;
                 letter-spacing: -0.01em; margin: 6px 0 4px 0; }
.section-sub { color: #7A6A58; font-size: 0.95rem; margin-bottom: 18px; }

/* recommendation card */
.rec-card { background: linear-gradient(180deg, #FFFDF8 0%, #FFF9EF 100%); border: 1px solid #EBDCC3; border-radius: 18px;
            padding: 22px 24px; margin-bottom: 16px;
            box-shadow: 0 4px 16px rgba(122, 52, 22, 0.07); }
.rec-top { display: flex; align-items: center; gap: 14px; }
.rank-badge { flex: 0 0 auto; width: 44px; height: 44px; border-radius: 12px;
              display: flex; align-items: center; justify-content: center;
              font-size: 1.25rem; font-weight: 800; color: #FFF8EC;
              background: linear-gradient(135deg, #2E1A0E 0%, #8C3A1B 55%, #DE8A2E 100%); }
.role-name { font-family: 'Fraunces', Georgia, serif; font-size: 1.3rem; font-weight: 700; color: #2E1A0E; line-height: 1.2; }
.role-sub { font-size: 0.86rem; color: #7A6A58; margin-top: 2px; }
.score { margin-left: auto; text-align: right; }
.score-num { font-size: 1.6rem; font-weight: 800; color: #8C3A1B; line-height: 1; }
.score-lbl { font-size: 0.72rem; color: #A89880; font-weight: 600;
             letter-spacing: 0.06em; text-transform: uppercase; margin-top: 3px; }

/* match bar */
.bar-track { height: 8px; background: #F1E4D3; border-radius: 999px;
             margin: 16px 0 14px 0; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 999px;
            background: linear-gradient(90deg, #DE8A2E, #C05A1E 55%, #8C3A1B); }

/* skill chips */
.chip-row { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px; }
.chip-label { font-size: 0.78rem; font-weight: 700; color: #A89880;
              text-transform: uppercase; letter-spacing: 0.05em; margin-right: 4px; }
.chip-have { background: #E7F2E4; color: #2F6B3A; border: 1px solid #BFDDB4;
             border-radius: 999px; padding: 4px 12px; font-size: 0.85rem; font-weight: 600; }
.chip-learn { background: #FDF0D5; color: #92400E; border: 1px solid #F3D9A4;
              border-radius: 999px; padding: 4px 12px; font-size: 0.85rem; font-weight: 600; }

/* empty state */
.empty-box { background: linear-gradient(180deg, #FFFDF8, #FBF3E4); border: 1.5px dashed #D9C7A7; border-radius: 18px;
             padding: 34px 28px; text-align: center; color: #7A6A58; font-size: 1rem; }
.empty-box b { color: #2E1A0E; }

/* sidebar — warm vertical gradient */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #FBF3E4 0%, #F6E7CE 100%); }
section[data-testid="stSidebar"] h2 { color: #2E1A0E; font-family: 'Fraunces', Georgia, serif; }

/* primary button — warm gradient */
.stButton > button { background: linear-gradient(120deg, #8C3A1B, #DE8A2E); color: #FFF8EC !important;
                     border: none; font-weight: 700; border-radius: 999px; }
.stButton > button:hover { background: linear-gradient(120deg, #7A3416, #C05A1E); color: #FFF8EC !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- hero banner
st.markdown(
    """
<div class="hero">
  <h1>🧭 Tech Stack Recommender</h1>
  <p>Tell me three skills you have. The Matchmaker Engine maps them to real job
  roles with TF-IDF + cosine similarity and returns your Top 3 career paths —
  with the skills you already have, and the ones still to learn.</p>
  <span class="stat">📋 12 job roles</span>
  <span class="stat">🛠️ 34 skills</span>
  <span class="stat">⚖️ TF-IDF + cosine similarity</span>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------ load the dataset
# Resolve the CSV relative to this file, not the working directory:
# Streamlit Cloud runs the app from the repo root, so a bare
# "raw_skills.csv" would look in the wrong folder.
DATA_PATH = Path(__file__).parent / "raw_skills.csv"
df = pd.read_csv(DATA_PATH)
df["skill_list"] = df["skills"].apply(lambda s: [x.strip() for x in s.split("|")])

# --------------------------- TF-IDF: each role is one document made of its skills
docs = ["|".join(skills) for skills in df["skill_list"]]
vectorizer = TfidfVectorizer(
    tokenizer=lambda d: d.split("|"),
    preprocessor=lambda d: d,
    lowercase=False,
    token_pattern=None,
)
tfidf_matrix = vectorizer.fit_transform(docs)

# ------------------------------------------------ sidebar: pick exactly 3 skills
all_skills = sorted(vectorizer.get_feature_names_out())
st.sidebar.header("Your skills")
if "picked" not in st.session_state:
    st.session_state.picked = []
if st.sidebar.button("✨ Try an example"):
    st.session_state.picked = ["Python", "Cloud Computing", "Automation"]
picked = st.sidebar.multiselect(
    "Pick exactly 3 skills you have:",
    options=all_skills,
    max_selections=3,
    key="picked",
)
st.sidebar.caption(
    "Rare skills count more: knowing Kubernetes points at DevOps far more "
    "sharply than knowing Python points at anything."
)

# ------------------------------------------------------------- recommendations
st.markdown('<div class="section-title">Your Top 3 career paths</div>', unsafe_allow_html=True)

if len(picked) < 3:
    st.markdown(
        f"""<div class="empty-box">
        <b>Pick 3 skills to see your matches</b> &nbsp;·&nbsp; {len(picked)}/3 selected<br>
        Use the sidebar on the left, or hit <b>✨ Try an example</b> to see it work instantly.
        </div>""",
        unsafe_allow_html=True,
    )
else:
    user_vec = vectorizer.transform(["|".join(picked)])
    scores = cosine_similarity(user_vec, tfidf_matrix)[0]
    out = df.copy()
    out["score"] = np.round(scores, 3)
    out["matched"] = out["skill_list"].apply(lambda gs: [g for g in gs if g in picked])
    out["to_learn"] = out["skill_list"].apply(lambda gs: [g for g in gs if g not in picked])
    top3 = out.sort_values("score", ascending=False).head(3)

    st.markdown(
        f'<div class="section-sub">Ranked by similarity to your 3 skills '
        f'· {len(df)} roles checked</div>',
        unsafe_allow_html=True,
    )
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        have_chips = "".join(f'<span class="chip-have">{s}</span>' for s in row["matched"])
        learn_chips = "".join(f'<span class="chip-learn">{s}</span>' for s in row["to_learn"])
        st.markdown(
            f"""
<div class="rec-card">
  <div class="rec-top">
    <div class="rank-badge">{rank}</div>
    <div>
      <div class="role-name">{row["role"]}</div>
      <div class="role-sub">{len(row["matched"])} of {len(row["skill_list"])} skills matched</div>
    </div>
    <div class="score">
      <div class="score-num">{row["score"]:.3f}</div>
      <div class="score-lbl">match score</div>
    </div>
  </div>
  <div class="bar-track"><div class="bar-fill" style="width: {float(row["score"]) * 100:.1f}%"></div></div>
  <div class="chip-row"><span class="chip-label">You have</span>{have_chips or '<span style="color:#9CA3AF">—</span>'}</div>
  <div class="chip-row"><span class="chip-label">To learn</span>{learn_chips}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("See all roles ranked"):
        st.dataframe(
            out.sort_values("score", ascending=False)[["role", "skills", "score"]],
            use_container_width=True,
        )

# ------------------------------------------------------------------- explainers
with st.expander("How does it work?"):
    st.write(
        """
        1. Each job role is treated as a document made of its skills (from raw_skills.csv).
        2. TF-IDF turns every role into a vector: a skill counts more when it is rare across roles (Kubernetes) and less when it is everywhere (Python).
        3. Your 3 skills become a vector with the same weights.
        4. Cosine similarity measures the angle between your vector and each role's vector: 1.0 = a perfect match, 0 = nothing in common.
        5. The Top 3 roles are shown, with the skills you already have and the ones still to learn.
        """
    )

with st.expander("See the dataset (raw_skills.csv)"):
    st.dataframe(df[["role", "skills"]], use_container_width=True)

st.divider()
st.caption("Tech Stack Recommender · TF-IDF + cosine similarity · Decode Labs Project 3 Capstone")
