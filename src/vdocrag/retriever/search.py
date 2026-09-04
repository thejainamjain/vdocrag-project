"""
Top-k retrieval given saved query and corpus embeddings (from encode.py).

Uses exact cosine similarity via a matrix multiply for small/medium corpora
(fine for a course-project-scale OpenDocVQA subset). For a "scalable" story,
swap `exact_search` for `faiss_search` once the corpus grows past a few
hundred-thousand vectors -- that swap is a natural extension point too.

Usage:
    python -m vdocrag.retriever.search \
        --query_embeddings outputs/embeddings/query.pkl \
        --corpus_embeddings outputs/embeddings/corpus.pkl \
        --top_k 10 \
        --output_path outputs/embeddings/rank.txt
"""

import argparse
import pickle

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query_embeddings", type=str, required=True)
    p.add_argument("--corpus_embeddings", type=str, required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--use_faiss", action="store_true",
                    help="Use FAISS approximate search instead of exact matmul (for large corpora)")
    return p.parse_args()


def exact_search(query_reps: np.ndarray, corpus_reps: np.ndarray, top_k: int):
    scores = query_reps @ corpus_reps.T  # (num_queries, num_docs), embeddings are already L2-normalized
    top_k = min(top_k, scores.shape[1])
    top_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]
    # sort the top_k by actual score, descending
    row_idx = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row_idx, top_idx], axis=1)
    top_idx = top_idx[row_idx, order]
    top_scores = scores[row_idx, top_idx]
    return top_idx, top_scores


def faiss_search(query_reps: np.ndarray, corpus_reps: np.ndarray, top_k: int):
    import faiss
    dim = corpus_reps.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product == cosine sim since vectors are normalized
    index.add(corpus_reps.astype(np.float32))
    scores, top_idx = index.search(query_reps.astype(np.float32), top_k)
    return top_idx, scores


def main():
    args = parse_args()

    query_ids, query_reps = pickle.load(open(args.query_embeddings, "rb"))
    corpus_ids, corpus_reps = pickle.load(open(args.corpus_embeddings, "rb"))

    search_fn = faiss_search if args.use_faiss else exact_search
    top_idx, top_scores = search_fn(query_reps, corpus_reps, args.top_k)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for qi, qid in enumerate(query_ids):
            for rank, (di, score) in enumerate(zip(top_idx[qi], top_scores[qi])):
                f.write(f"{qid}\t{corpus_ids[di]}\t{rank + 1}\t{score:.6f}\n")

    print(f"Wrote rankings for {len(query_ids)} queries to {args.output_path}")


if __name__ == "__main__":
    main()
