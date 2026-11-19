# Tech Stack Recommender — Decode Labs Project 3 (Capstone)

Pick 3 skills you have. The Matchmaker Engine maps them to real job roles with
TF-IDF + cosine similarity and returns your **Top 3 career paths** — with the
skills you already have, and the ones still to learn.

Part of the Decode Labs AI Internship (27 Oct – 27 Nov 2026). Project 3 is the
capstone; Project 4 (Image/Text Recognition with Tesseract OCR) is in
`Project-4-Image-Text-Recognition/`.

## How it works

1. Each job role is treated as a document made of its skills (`raw_skills.csv`).
2. TF-IDF turns every role into a vector: a skill counts more when it is rare
   across roles (Kubernetes) and less when it is everywhere (Python).
3. Your 3 skills become a vector with the same weights.
4. Cosine similarity measures the angle between your vector and each role's
   vector: 1.0 = a perfect match, 0 = nothing in common.
5. The Top 3 roles are shown with a match score, the skills you already have
   and the ones still to learn. All 12 roles ranked are available in an
   expandable table, alongside the full dataset.

## Dataset

12 roles, 34 unique skills (`raw_skills.csv`):

| Role | Skills |
| --- | --- |
| Data Scientist | Python, SQL, Machine Learning, Data Analysis, Statistics |
| Machine Learning Engineer | Python, Machine Learning, Deep Learning, TensorFlow, MLOps |
| DevOps Engineer | AWS, Docker, Kubernetes, CI/CD, Linux |
| Cloud Architect | AWS, Cloud Computing, Networking, Security, Terraform |
| Backend Developer | Java, Python, SQL, APIs, Data Structures |
| Frontend Developer | JavaScript, React, HTML, CSS, TypeScript |
| Full Stack Developer | JavaScript, Python, React, SQL, APIs |
| Data Engineer | Python, SQL, Spark, ETL, Cloud Computing |
| Cybersecurity Analyst | Networking, Security, Linux, Python, Cryptography |
| Mobile Developer | Kotlin, Swift, Java, APIs, UI/UX |
| Systems Administrator | Linux, Networking, Automation, Security, Scripting |
| AI Engineer | Python, Machine Learning, Deep Learning, NLP, APIs |

## Try it

Pick exactly 3 skills in the sidebar, or hit **✨ Try an example**
(Python + Cloud Computing + Automation) to see it work instantly.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

Repo `Sharjeelsarwar-ai2/decode-labs-internship`, branch `main`, main file path
`Project-3-Recommendation-System/app.py`. No `packages.txt` needed — this app
uses no system packages.

## Stack

Streamlit, pandas, NumPy, scikit-learn (TF-IDF + cosine similarity).

## Links

- LinkedIn walkthrough video: https://www.linkedin.com/feed/update/urn:li:activity:7527324159429931009/
- Code: https://github.com/Sharjeelsarwar-ai2/decode-labs-internship
