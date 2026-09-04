"""
Downloads a small, manageable subset of OpenDocVQA + its corpus from
Hugging Face, and converts it into the JSONL formats our modules expect
(retriever_train.jsonl, corpus.jsonl, test_queries.jsonl, generator_train.jsonl).

We deliberately pull just 1-2 sub-datasets (e.g. ChartQA + SlideVQA) instead
of the full 9-dataset / 200k-image OpenDocVQA, since a full download/training
run isn't necessary (or feasible) for a course project.

Usage:
    python data/download_opendocvqa.py --configs chartqa slidevqa --max_examples 500
"""

import argparse
import json
import os

from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="+", default=["chartqa"],
                    help="Which OpenDocVQA sub-datasets to pull, e.g. chartqa slidevqa infovqa dude")
    p.add_argument("--max_examples", type=int, default=500,
                    help="Cap per-config examples so downloads/training stay small")
    p.add_argument("--output_dir", type=str, default="data/processed")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    images_dir = os.path.join(args.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    retriever_rows, corpus_rows, query_rows, generator_rows = [], [], [], []
    corpus_seen = set()

    for config in args.configs:
        print(f"Loading QA pairs: NTT-hil-insight/OpenDocVQA [{config}] ...")
        qa_ds = load_dataset("NTT-hil-insight/OpenDocVQA", config, split="test")
        qa_ds = qa_ds.select(range(min(args.max_examples, len(qa_ds))))

        print(f"Loading corpus: NTT-hil-insight/OpenDocVQA-Corpus [{config}] ...")
        corpus_ds = load_dataset("NTT-hil-insight/OpenDocVQA-Corpus", config, split="test")

        for i, ex in enumerate(corpus_ds):
            doc_id = f"{config}_{ex.get('image_id', i)}"
            if doc_id in corpus_seen:
                continue
            corpus_seen.add(doc_id)
            rel_path = f"{config}_{i}.png"
            ex["image"].save(os.path.join(images_dir, rel_path))
            corpus_rows.append({"id": doc_id, "image": rel_path})

        for i, ex in enumerate(qa_ds):
            qid = f"{config}_q{i}"
            query_rows.append({"id": qid, "query": ex["question"]})

            # NOTE: field names vary slightly per source dataset; adjust the
            # keys below (`image_id`, `answer`) after inspecting one example
            # with `print(qa_ds[0])` if this errors on a given config.
            pos_doc_id = f"{config}_{ex.get('image_id', 0)}"
            answer = ex.get("answers", [ex.get("answer", "")])
            answer = answer[0] if isinstance(answer, list) else answer

            retriever_rows.append({
                "query": ex["question"],
                "pos_image": next((c["image"] for c in corpus_rows if c["id"] == pos_doc_id), None),
            })
            generator_rows.append({
                "query": ex["question"],
                "context_images": [next((c["image"] for c in corpus_rows if c["id"] == pos_doc_id), None)],
                "answer": answer,
            })

    def dump(rows, name):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                if all(v is not None for v in r.values()):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {path}")

    dump(corpus_rows, "corpus.jsonl")
    dump(query_rows, "test_queries.jsonl")
    dump(retriever_rows, "retriever_train.jsonl")
    dump(generator_rows, "generator_train.jsonl")

    print("Done. NOTE: inspect the JSONL files -- source dataset field names "
          "differ slightly between ChartQA/SlideVQA/InfoVQA/DUDE, so double "
          "check 'answer' and image-id matching before training.")


if __name__ == "__main__":
    main()
