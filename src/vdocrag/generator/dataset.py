"""
Dataset for generator SFT.

Expected JSONL format, one example per line:
    {
      "query": "How many international visitors came to Japan in 2017?",
      "context_images": ["page_23.png"],   # gold or retrieved page(s), top_k already applied upstream
      "answer": "28.69m"
    }
"""

import json
from typing import List, Dict, Any

from PIL import Image
from torch.utils.data import Dataset


class GeneratorSFTDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str = ""):
        self.examples: List[Dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.examples.append(json.loads(line))
        self.image_root = image_root

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        images = []
        for rel_path in ex["context_images"]:
            path = f"{self.image_root}/{rel_path}" if self.image_root else rel_path
            images.append(Image.open(path).convert("RGB"))
        return {"question": ex["query"], "images": images, "answer": ex["answer"]}


def generator_collate_fn(batch, processor, max_len: int = 1024):
    """
    Builds chat-formatted (prompt + gold answer) sequences and masks the loss
    so only the answer tokens contribute to the SFT loss (prompt tokens are
    ignored, labels = -100).
    """
    input_texts, all_images, answer_lens = [], [], []

    for ex in batch:
        image_tokens = "\n".join([f"<|image_{i + 1}|>" for i in range(len(ex["images"]))])
        messages = [{"role": "user", "content": f"{image_tokens}\n{ex['question']}\nAnswer briefly."}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = prompt + ex["answer"] + processor.tokenizer.eos_token

        input_texts.append(full_text)
        all_images.append(ex["images"])
        answer_lens.append(len(processor.tokenizer(ex["answer"])["input_ids"]))

    inputs = processor(input_texts, images=all_images, return_tensors="pt",
                        padding="longest", truncation=True, max_length=max_len)

    labels = inputs["input_ids"].clone()
    labels[inputs["attention_mask"] == 0] = -100
    # Mask everything except the last `answer_len` tokens per example (prompt is not supervised).
    for i, ans_len in enumerate(answer_lens):
        labels[i, : -ans_len - 1] = -100  # -1 leaves room for the eos token

    inputs["labels"] = labels
    return inputs
