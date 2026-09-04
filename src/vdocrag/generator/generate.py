"""
End-to-end RAG inference: for each question, load its top-k retrieved page
images (rank file produced by retriever/search.py) and generate the answer.

Usage:
    python -m vdocrag.generator.generate \
        --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
        --lora_name_or_path outputs/generator-qwen2vl2b-lora \
        --queries_jsonl data/processed/test_queries.jsonl \
        --corpus_jsonl data/processed/corpus.jsonl \
        --image_root data/processed/images \
        --rank_file outputs/embeddings/rank.txt \
        --top_k 3 \
        --output_path outputs/answers/answers.json
"""

import argparse
import json
from collections import defaultdict

from PIL import Image
from tqdm import tqdm

from vdocrag.generator.modeling import VDocGenerator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--lora_name_or_path", type=str, default=None)
    p.add_argument("--queries_jsonl", type=str, required=True)
    p.add_argument("--corpus_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, default="")
    p.add_argument("--rank_file", type=str, required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output_path", type=str, required=True)
    return p.parse_args()


def load_rankings(rank_file: str, top_k: int):
    rankings = defaultdict(list)
    with open(rank_file, encoding="utf-8") as f:
        for line in f:
            qid, docid, rank, score = line.strip().split("\t")
            if int(rank) <= top_k:
                rankings[qid].append(docid)
    return rankings


def main():
    args = parse_args()

    queries = {json.loads(l)["id"]: json.loads(l)["query"]
               for l in open(args.queries_jsonl, encoding="utf-8") if l.strip()}
    corpus = {json.loads(l)["id"]: json.loads(l)["image"]
              for l in open(args.corpus_jsonl, encoding="utf-8") if l.strip()}
    rankings = load_rankings(args.rank_file, args.top_k)

    generator = VDocGenerator.load(
        args.model_name_or_path, lora_name_or_path=args.lora_name_or_path,
        use_lora=args.lora_name_or_path is not None,
    ).eval()

    results = {}
    for qid, question in tqdm(queries.items()):
        doc_ids = rankings.get(qid, [])[:args.top_k]
        images = []
        for did in doc_ids:
            rel_path = corpus[did]
            path = f"{args.image_root}/{rel_path}" if args.image_root else rel_path
            images.append(Image.open(path).convert("RGB"))

        if not images:
            results[qid] = {"question": question, "answer": "", "retrieved": []}
            continue

        answer = generator.answer(question + "\nAnswer briefly.", images)
        results[qid] = {"question": question, "answer": answer, "retrieved": doc_ids}

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} answers to {args.output_path}")


if __name__ == "__main__":
    main()
