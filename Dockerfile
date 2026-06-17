FROM python:3.11-slim

WORKDIR /app

# Core dependencies
RUN pip install --no-cache-dir "numpy>=1.24.0" "sentence-transformers>=2.6.0"

COPY ranker/       ./ranker/
COPY rank.py       ./
COPY validate_submission.py ./

# Default: reproduce the submission (hybrid TF-IDF + neural top 500, ~153 s)
# docker build -t redrob-ranker .
# docker run -v $(pwd)/candidates.jsonl:/app/candidates.jsonl redrob-ranker
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./purvi-porwal.csv", "--verbose"]

# ── Fallback (TF-IDF only, no sentence-transformers) ─────────────────────────
# CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./purvi-porwal.csv", "--tfidf"]

# ── Sandbox ───────────────────────────────────────────────────────────────────
# docker run -p 8501:8501 redrob-ranker \
#   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
