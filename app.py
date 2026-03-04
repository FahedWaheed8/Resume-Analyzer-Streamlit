import re
import streamlit as st
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ — Keyword Analyzer",
    page_icon="📄",
    layout="wide"
)

# ─── Tech Keyword Bank ────────────────────────────────────────────────────────
TECH_KEYWORDS = {
    "Languages": [
        "python", "javascript", "typescript", "java", "c++", "c#", "go", "rust",
        "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql", "bash", "html", "css"
    ],
    "Frameworks & Libraries": [
        "react", "vue", "angular", "django", "flask", "fastapi", "spring", "express",
        "next.js", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
        "spark", "kafka", "graphql", "rest", "grpc", "langchain"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "jenkins", "github actions", "circleci", "helm", "prometheus", "nginx",
        "linux", "ci/cd", "devops", "microservices"
    ],
    "Databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "sqlite", "oracle", "neo4j", "firebase", "snowflake", "bigquery"
    ],
    "Concepts": [
        "machine learning", "deep learning", "nlp", "computer vision", "data science",
        "api", "agile", "scrum", "tdd", "oop", "distributed systems", "system design",
        "algorithms", "data structures", "security", "oauth", "jwt", "caching",
        "load balancing", "etl", "data pipeline"
    ],
    "Tools": [
        "git", "jira", "confluence", "figma", "tableau", "power bi",
        "airflow", "dbt", "mlflow", "hugging face", "openai"
    ]
}


# ─── Helper Functions ─────────────────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages).strip()


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def extract_keywords(text: str) -> dict:
    text_lower = text.lower()
    found = {}
    for category, keywords in TECH_KEYWORDS.items():
        matches = [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', text_lower)]
        if matches:
            found[category] = matches
    return found


def compute_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
    matrix = vectorizer.fit_transform([clean_text(resume_text), clean_text(jd_text)])
    return round(float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0]) * 100, 1)


def get_top_tfidf_terms(text: str, top_n: int = 20) -> list:
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=1000)
    try:
        matrix = vectorizer.fit_transform([clean_text(text)])
        feature_names = vectorizer.get_feature_names_out()
        scores = matrix.toarray()[0]
        top_indices = scores.argsort()[-top_n:][::-1]
        return [feature_names[i] for i in top_indices if scores[i] > 0]
    except Exception:
        return []


def analyze(resume_text: str, jd_text: str) -> dict:
    similarity = compute_similarity(resume_text, jd_text)
    resume_kw = extract_keywords(resume_text)
    jd_kw = extract_keywords(jd_text)

    resume_flat = {kw for kws in resume_kw.values() for kw in kws}
    jd_flat = {kw for kws in jd_kw.values() for kw in kws}

    matched = sorted(resume_flat & jd_flat)
    missing = sorted(jd_flat - resume_flat)

    jd_top = get_top_tfidf_terms(jd_text, 30)
    resume_top = set(get_top_tfidf_terms(resume_text, 50))
    missing_tfidf = [t for t in jd_top if t not in resume_top][:15]

    missing_by_cat = {
        cat: [kw for kw in kws if kw in missing]
        for cat, kws in TECH_KEYWORDS.items()
        if any(kw in missing for kw in kws)
    }

    kw_match_pct = round(len(matched) / len(jd_flat) * 100, 1) if jd_flat else 0.0
    overall = round(similarity * 0.6 + kw_match_pct * 0.4, 1)

    return {
        "overall": overall,
        "similarity": similarity,
        "kw_match": kw_match_pct,
        "matched": matched,
        "missing": missing,
        "missing_by_cat": missing_by_cat,
        "missing_tfidf": missing_tfidf,
    }


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("📄 ResumeIQ — Smart Keyword Analyzer")
st.caption("Upload your resume and a job description to get an instant match analysis.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("🟣 Your Resume")
    resume_file = st.file_uploader("Upload PDF", type=["pdf"], key="resume")
    resume_text_input = st.text_area("Or paste resume text", height=200, placeholder="Paste your resume here...")

with col2:
    st.subheader("🔵 Job Description")
    jd_file = st.file_uploader("Upload PDF", type=["pdf"], key="jd")
    jd_text_input = st.text_area("Or paste job description", height=200, placeholder="Paste the job description here...")

st.divider()

if st.button("⚡ Analyze Match", use_container_width=True, type="primary"):

    # Get text
    resume_text = extract_text_from_pdf(resume_file) if resume_file else resume_text_input.strip()
    jd_text = extract_text_from_pdf(jd_file) if jd_file else jd_text_input.strip()

    if not resume_text:
        st.error("Please provide your resume (upload or paste text).")
    elif not jd_text:
        st.error("Please provide the job description (upload or paste text).")
    else:
        with st.spinner("Analyzing..."):
            result = analyze(resume_text, jd_text)

        st.divider()
        st.subheader("📊 Results")

        # ── Verdict ──
        overall = result["overall"]
        if overall >= 60:
            st.success(f"🚀 Strong Match! Your resume aligns well with this job.")
        elif overall >= 35:
            st.warning(f"⚡ Moderate Match. Add the missing keywords to improve your chances.")
        else:
            st.error(f"⚠️ Low Match. Your resume needs significant tailoring for this role.")

        # ── Score Metrics ──
        m1, m2, m3 = st.columns(3)
        m1.metric("Overall Score", f"{result['overall']}%")
        m2.metric("Cosine Similarity", f"{result['similarity']}%")
        m3.metric("Keyword Match", f"{result['kw_match']}%")

        # ── Progress Bars ──
        st.write("**Overall Score**")
        st.progress(result["overall"] / 100)
        st.write("**Cosine Similarity**")
        st.progress(result["similarity"] / 100)
        st.write("**Keyword Match**")
        st.progress(result["kw_match"] / 100)

        st.divider()

        # ── Matched & Missing ──
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("✅ Matched Keywords")
            if result["matched"]:
                st.write(" ".join([f"`{kw}`" for kw in result["matched"]]))
            else:
                st.write("_No technical keywords matched._")

        with col_b:
            st.subheader("❌ Missing Keywords")
            if result["missing_by_cat"]:
                for cat, kws in result["missing_by_cat"].items():
                    st.markdown(f"**{cat}**")
                    st.write(" ".join([f"`{kw}`" for kw in kws]))
            else:
                st.write("_No critical missing keywords! 🎉_")

        st.divider()

        # ── TF-IDF Gap Terms ──
        st.subheader("🔍 Top JD Terms Not In Resume")
        st.caption("Identified via TF-IDF — consider weaving these into your resume.")
        if result["missing_tfidf"]:
            st.write(" ".join([f"`{t}`" for t in result["missing_tfidf"]]))
        else:
            st.write("_No significant term gaps found._")
