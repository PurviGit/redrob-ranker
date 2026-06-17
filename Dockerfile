FROM python:3.11-slim

WORKDIR /app

# Core dependencies
RUN pip install --no-cache-dir "numpy>=1.24.0" "sentence-transformers>=2.6.0"

# Pre-download both models at build time so Docker run works fully offline
# bi-encoder (~80 MB) + cross-encoder (~67 MB) cached in HF_HOME
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
from sentence_transformers.cross_encoder import CrossEncoder; \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached.')"

COPY ranker/       ./ranker/
COPY rank.py       ./
COPY validate_submission.py ./

# Default: reproduce the submission (3-stage: TF-IDF + bi-encoder top 500 + cross-encoder top 30, ~177 s)
# docker build -t redrob-ranker .
# docker run -v $(pwd)/candidates.jsonl:/app/candidates.jsonl redrob-ranker
CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./purvi-porwal.csv", "--verbose"]

# ── Fallback (TF-IDF only, no sentence-transformers) ─────────────────────────
# CMD ["python", "rank.py", "--candidates", "./candidates.jsonl", "--out", "./purvi-porwal.csv", "--tfidf"]

# ── Sandbox ───────────────────────────────────────────────────────────────────
# docker run -p 8501:8501 redrob-ranker \
#   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
