# ResumeIQ — Smart Resume Keyword Analyzer

> **Resume bullet**: Built an NLP-based resume analyzer using Python and scikit-learn to compute TF-IDF cosine similarity scores against job descriptions.

## What it does

Upload your resume and a job description → get an instant match analysis with keyword insights.

- 📄 Upload resume or job description as **PDF or plain text**
- 🧠 Computes **TF-IDF cosine similarity** score between resume and JD
- 🏷️ Extracts and matches **100+ technical keywords** across 6 categories
- ❌ Highlights **missing skills** grouped by category
- 🔍 Shows **top JD terms not in resume** via TF-IDF
- 📊 Three-score dashboard: Overall, Cosine Similarity, Keyword Match %

## Tech Stack

| Layer | Tech |
|-------|------|
| Language | Python |
| UI | Streamlit |
| NLP / ML | scikit-learn (TF-IDF, cosine similarity) |
| PDF Parsing | pdfplumber |
| Version Control | Git / GitHub |

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/FahedWaheed8/Resume-Analyzer-Streamlit.git
cd Resume-Analyzer-Streamlit

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install streamlit pdfplumber scikit-learn numpy

# 4. Run the app
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

## How it Works

1. **Text Extraction** — `pdfplumber` extracts raw text from uploaded PDFs
2. **TF-IDF Vectorization** — both documents vectorized using `TfidfVectorizer` with 1-2 gram range
3. **Cosine Similarity** — measures similarity between resume and JD vectors (0–100%)
4. **Keyword Matching** — regex scan against 100+ tech keywords across 6 categories
5. **Overall Score** — weighted blend: 60% cosine similarity + 40% keyword match %
6. **Gap Analysis** — highlights keywords in JD but missing from resume

## Project Structure

```
Resume-Analyzer-Streamlit/
├── app.py              ← All logic and UI in pure Python
├── requirements.txt
└── README.md
```
