"""
Dataset classes for the retriever.

Expected data format (JSONL), one example per line:
    {
      "query": "How many international visitors came to Japan in 2017?",
      "pos_image": "path/or/hf-url/to/page_23.png",
      "neg_images": ["path/to/distractor_1.png", "path/to/distractor_2.png"]   # optional, hard negatives
    }

In-batch negatives are used by default (other examples' positives act as
negatives for a given query), matching VDocRAG's training setup. Hard
negatives (neg_images) are optional extras you can add later for a stronger
signal — this is also a natural extension point (hard-negative mining).
"""

import json
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset

QUERY_INSTRUCTION = "Instruct: I'm looking for an image that answers the question.\nQuery: {query}"
DOC_PROMPT = "<|image_1|>\nWhat is shown in this image?"


class RetrieverContrastiveDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str = ""):
        self.examples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.examples)

    def _load_image(self, rel_path: str) -> Image.Image:
        path = f"{self.image_root}/{rel_path}" if self.image_root else rel_path
        return Image.open(path).convert("RGB")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        query_text = QUERY_INSTRUCTION.format(query=ex["query"])
        pos_image = self._load_image(ex["pos_image"])
        return {
            "query_text": query_text,
            "doc_prompt": DOC_PROMPT,
            "doc_image": pos_image,
        }


def retriever_collate_fn(batch: List[Dict[str, Any]], processor, query_max_len: int = 256,
                          doc_max_len: int = 4096):
    """
    Collates a batch into model-ready tensors using the backbone's processor
    (works for both Phi-3-vision's and Qwen2-VL's AutoProcessor, since both
    expose the same __call__ signature for text+images).
    """
    queries = [b["query_text"] for b in batch]
    doc_prompts = [b["doc_prompt"] for b in batch]
    doc_images = [b["doc_image"] for b in batch]

    query_inputs = processor(queries, return_tensors="pt", padding="longest",
                              max_length=query_max_len, truncation=True)

    doc_inputs = processor(doc_prompts, images=doc_images, return_tensors="pt",
                            padding="longest", max_length=doc_max_len, truncation=True)

    return {"query": query_inputs, "document": doc_inputs}
