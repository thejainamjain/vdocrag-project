"""
Encode a query set or a document corpus into dense embeddings, saved as .pkl
(list of ids + a stacked numpy array), so retrieval (search.py) doesn't need
the model loaded again.

Usage:
    python -m vdocrag.retriever.encode \
        --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
        --lora_name_or_path outputs/retriever-qwen2vl2b-lora \
        --input_jsonl data/processed/corpus.jsonl \
        --image_root data/processed/images \
        --mode document \
        --output_path outputs/embeddings/corpus.pkl
"""

import argparse
import json
import pickle

import torch
from PIL import Image
from tqdm import tqdm

from vdocrag.retriever.modeling import VDocRetriever
from vdocrag.retriever.dataset import QUERY_INSTRUCTION, DOC_PROMPT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--lora_name_or_path", type=str, default=None)
    p.add_argument("--input_jsonl", type=str, required=True,
                    help="JSONL with either {'id','query'} rows (mode=query) "
                         "or {'id','image'} rows (mode=document)")
    p.add_argument("--image_root", type=str, default="")
    p.add_argument("--mode", choices=["query", "document"], required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = VDocRetriever.load(
        args.model_name_or_path, lora_name_or_path=args.lora_name_or_path,
        use_lora=args.lora_name_or_path is not None,
    ).eval().to("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(args.input_jsonl, encoding="utf-8") if l.strip()]

    ids, reps = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch_size)):
            batch = rows[i:i + args.batch_size]
            ids.extend([r["id"] for r in batch])

            if args.mode == "query":
                texts = [QUERY_INSTRUCTION.format(query=r["query"]) for r in batch]
                inputs = processor(texts, return_tensors="pt", padding="longest",
                                    max_length=256, truncation=True).to(model.base_model.device)
                out = model.encode_query(inputs)
            else:
                images = [Image.open(f"{args.image_root}/{r['image']}" if args.image_root else r["image"]).convert("RGB")
                          for r in batch]
                prompts = [DOC_PROMPT] * len(batch)
                inputs = processor(prompts, images=images, return_tensors="pt",
                                    padding="longest", max_length=2048, truncation=True).to(model.base_model.device)
                out = model.encode_document(inputs)

            reps.append(out.float().cpu())

    reps = torch.cat(reps, dim=0).numpy()
    with open(args.output_path, "wb") as f:
        pickle.dump((ids, reps), f)
    print(f"Saved {len(ids)} embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
