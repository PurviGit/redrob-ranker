"""
ranker/semantic.py  —  Semantic similarity engine

Mode: Hybrid (default)
  Step 1: TF-IDF on all ~32K survivors  (~40 s, ~200 MB RAM)
  Step 2: Neural re-score top 2000 by TF-IDF using all-MiniLM-L6-v2  (~60 s)
  Step 3: Merge — neural score where available, TF-IDF elsewhere
  Total: ~100 s, well within 5-min compute budget.

Why not neural on all 32K? Encoding 32K texts with all-MiniLM-L6-v2 on CPU
takes 10+ minutes — violates the 5-minute constraint. Neural on top 2000
gives quality where it matters (the candidates actually competing for top 100)
without blowing the budget.

Fallback (--tfidf flag): pure TF-IDF on all candidates, ~40 s.
"""
from __future__ import annotations
import math
import re
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Optional

JD_SEMANTIC_DOCUMENT = """
Senior AI Engineer at a Series A AI-native talent intelligence platform.
Building the intelligence layer: ranking, retrieval, and candidate-job matching systems.
Production experience with embeddings-based retrieval systems deployed to real users.
Handled embedding drift, index refresh, retrieval-quality regression in production.
Vector databases: Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Hybrid search combining dense embeddings with BM25 sparse retrieval.
Evaluation frameworks for ranking systems: NDCG, MRR, MAP, offline-to-online correlation, A/B testing.
LLM-based re-ranking, fine-tuning transformer models, RAG pipelines.
Strong Python production code, not just notebooks or scripts.
Scrappy product-engineering attitude, shipping working systems quickly.
Migration from keyword-based search to embedding-based retrieval.
Learning to rank, XGBoost LightGBM ranking models, semantic search.
Information retrieval, relevance labeling, feature pipeline, training and evaluation.
Not pure research — production deployment experience required.
Pre-LLM retrieval experience valued highly.
"""

STRONG_SEMANTIC_PHRASES = [
    "embedding-based retrieval", "vector search", "semantic search",
    "ranking system", "retrieval system", "hybrid search", "dense retrieval",
    "learning to rank", "offline evaluation", "a/b testing", "ndcg", "mrr",
    "embedding drift", "index refresh", "retrieval quality",
    "production retrieval", "relevance labeling", "migrated keyword",
    "bm25", "ltr", "vector database", "recall@k",
    "fine-tuning llm", "rag pipeline", "two-stage retrieval",
]

# Number of top TF-IDF candidates to re-score with neural
NEURAL_RERANK_TOP_N = 500


def _tok(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def build_candidate_semantic_text(c: dict) -> str:
    p     = c.get("profile", {})
    parts = []
    s = p.get("summary", "")
    if s: parts.extend([s, s, s])
    h = p.get("headline", "")
    if h: parts.extend([h, h])
    for job in c.get("career_history", []):
        desc  = job.get("description", "")
        title = job.get("title", "")
        if desc:  parts.extend([desc, desc])
        if title: parts.append(title)
    sk_str = " ".join(sk["name"] for sk in c.get("skills", []))
    if sk_str: parts.append(sk_str)
    ct = p.get("current_title", "")
    if ct: parts.append(ct)
    return " ".join(parts)


def _embed_vec(tokens: list[str], vocab: dict, idf: np.ndarray) -> np.ndarray:
    V   = len(vocab)
    vec = np.zeros(V, dtype=np.float32)
    tot = max(len(tokens), 1)
    cnt = Counter(tokens)
    for t, n in cnt.items():
        if t in vocab:
            vec[vocab[t]] = n / tot
    vec *= idf
    nrm  = np.linalg.norm(vec)
    if nrm > 0:
        vec /= nrm
    return vec


class SemanticScorer:
    def __init__(self, use_neural: bool = True, precomputed_path: Optional[str] = None):
        self._model              = None
        self._use_neural         = False
        self._vocab: Optional[dict]       = None
        self._idf:   Optional[np.ndarray] = None
        self._jd_embedding: Optional[np.ndarray]     = None
        self._precomputed_scores: Optional[np.ndarray] = None
        self._candidate_ids: Optional[list]            = None
        self._precomputed_path = precomputed_path

        if use_neural:
            self._try_load_neural()
        if not self._use_neural:
            print("  [Semantic] TF-IDF mode (build+score ~90 s, ~300 MB RAM)")

    def _try_load_neural(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model      = SentenceTransformer("all-MiniLM-L6-v2")
            self._use_neural = True
            print(f"  [Semantic] Hybrid: TF-IDF all + neural top {NEURAL_RERANK_TOP_N} "
                  f"(all-MiniLM-L6-v2 {self._model.get_sentence_embedding_dimension()}d)")
        except Exception as e:
            print(f"  [Semantic] Neural unavailable ({e.__class__.__name__}), using TF-IDF only")
            self._use_neural = False

    def fit(self, candidates: list[dict]):
        print(f"  [Semantic] Building index for {len(candidates):,} candidates …")
        if self._precomputed_path and Path(self._precomputed_path).exists():
            self._load_precomputed(candidates)
            return

        self._candidate_ids = [c["candidate_id"] for c in candidates]
        if self._use_neural:
            self._fit_hybrid(candidates)
        else:
            self._fit_tfidf(candidates)

        lo = self._precomputed_scores.min()
        hi = self._precomputed_scores.max()
        print(f"  [Semantic] Done. Score range: {lo:.3f}–{hi:.3f}")
        if self._precomputed_path:
            self._save_precomputed()

    def _fit_tfidf(self, candidates: list[dict]) -> np.ndarray:
        """TF-IDF on all candidates. Returns score array and caches vocab/idf."""
        N = len(candidates)
        texts     = [build_candidate_semantic_text(c) for c in candidates]
        tok_lists = [_tok(t) for t in texts]
        del texts

        word_df: dict[str, int] = {}
        for toks in tok_lists:
            for t in set(toks):
                word_df[t] = word_df.get(t, 0) + 1

        top2k       = sorted(word_df.items(), key=lambda x: -x[1])[:2000]
        self._vocab = {w: i for i, (w, _) in enumerate(top2k)}
        V           = len(self._vocab)
        self._idf   = np.array(
            [math.log((N + 2) / (word_df.get(w, 0) + 1)) + 1.0 for w in self._vocab],
            dtype=np.float32,
        )
        jd_vec             = _embed_vec(_tok(JD_SEMANTIC_DOCUMENT), self._vocab, self._idf)
        self._jd_embedding = jd_vec
        del word_df

        CHUNK  = 10000
        scores = np.zeros(N, dtype=np.float32)
        for start in range(0, N, CHUNK):
            batch_toks = tok_lists[start : start + CHUNK]
            mat = np.zeros((len(batch_toks), V), dtype=np.float32)
            for i, toks in enumerate(batch_toks):
                tot = max(len(toks), 1)
                cnt = Counter(toks)
                for t, n in cnt.items():
                    if t in self._vocab:
                        mat[i, self._vocab[t]] = n / tot
            mat  *= self._idf
            nrms  = np.linalg.norm(mat, axis=1, keepdims=True)
            nrms[nrms == 0] = 1.0
            mat  /= nrms
            scores[start : start + len(batch_toks)] = mat @ jd_vec
            del mat

        self._precomputed_scores = np.clip(scores, 0, 1)
        return scores

    def _fit_hybrid(self, candidates: list[dict]):
        """TF-IDF on all → neural re-score top N → merge."""
        import time
        t0 = time.time()
        tfidf_scores = self._fit_tfidf(candidates)
        print(f"  [Semantic] TF-IDF done ({time.time()-t0:.1f}s). "
              f"Re-scoring top {NEURAL_RERANK_TOP_N} with neural …")

        # Indices of top N by TF-IDF
        top_idx = np.argsort(tfidf_scores)[-NEURAL_RERANK_TOP_N:]

        top_candidates = [candidates[i] for i in top_idx]
        texts = [build_candidate_semantic_text(c) for c in top_candidates]

        jd_emb = self._model.encode(
            [JD_SEMANTIC_DOCUMENT], normalize_embeddings=True)[0]
        self._jd_embedding = jd_emb

        neural_scores = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            embs  = self._model.encode(
                batch, normalize_embeddings=True,
                show_progress_bar=False, batch_size=64)
            neural_scores.extend((embs @ jd_emb).tolist())

        # Merge: replace TF-IDF scores for top N with neural scores
        merged = tfidf_scores.copy()
        for rank_i, orig_i in enumerate(top_idx):
            merged[orig_i] = float(np.clip(neural_scores[rank_i], 0, 1))

        self._precomputed_scores = merged
        print(f"  [Semantic] Neural re-rank done ({time.time()-t0:.1f}s total)")

    # ── public API ──────────────────────────────────────────────────────────
    def score_all(self) -> np.ndarray:
        if self._precomputed_scores is None:
            raise RuntimeError("Call fit() first")
        return self._precomputed_scores

    def score_single(self, candidate: dict) -> float:
        text = build_candidate_semantic_text(candidate)
        if self._use_neural and self._model:
            if self._jd_embedding is None:
                self._jd_embedding = self._model.encode(
                    [JD_SEMANTIC_DOCUMENT], normalize_embeddings=True)[0]
            emb = self._model.encode([text], normalize_embeddings=True)[0]
        else:
            if self._vocab is None:
                return 0.5
            emb = _embed_vec(_tok(text), self._vocab, self._idf)
        return float(np.clip(np.dot(emb, self._jd_embedding), 0, 1))

    def score_phrase_hits(self, candidate: dict) -> float:
        text = build_candidate_semantic_text(candidate).lower()
        hits = sum(1 for p in STRONG_SEMANTIC_PHRASES if p in text)
        return min(hits / 4.0, 1.0)

    def get_id_to_score_map(self) -> dict[str, float]:
        return dict(zip(self._candidate_ids, self.score_all().tolist()))

    def _save_precomputed(self):
        path = Path(self._precomputed_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), self._precomputed_scores)
        ids_path = str(path).replace(".npy", "_ids.txt")
        with open(ids_path, "w") as f:
            f.write("\n".join(self._candidate_ids))
        print(f"  [Semantic] Saved → {path}")

    def _load_precomputed(self, candidates: list[dict]):
        path = Path(self._precomputed_path)
        self._precomputed_scores = np.load(str(path))
        ids_path = str(path).replace(".npy", "_ids.txt")
        with open(ids_path) as f:
            self._candidate_ids = f.read().splitlines()
        print(f"  [Semantic] Loaded {len(self._precomputed_scores):,} precomputed scores")
