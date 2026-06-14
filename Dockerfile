FROM python:3.11-slim

WORKDIR /app

# Only numpy needed for ranking (+ stdlib)
RUN pip install --no-cache-dir "numpy>=1.24.0"

COPY ranker/       ./ranker/
COPY rank.py       ./
COPY validate_submission.py ./

# Default: reproduce the submission
# docker build -t redrob-ranker .
# docker run -v $(pwd)/candidates.jsonl:/app/candidates.jsonl redrob-ranker
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./submission.csv", "--tfidf", "--verbose"]

# ── Sandbox ───────────────────────────────────────────────────────────────────
# docker run -p 8501:8501 redrob-ranker \
#   streamlit run app.py --server.port 8501 --server.address 0.0.0.0

# ── Neural upgrade (optional) ─────────────────────────────────────────────────
# RUN pip install sentence-transformers
