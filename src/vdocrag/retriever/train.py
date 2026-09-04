"""
Fine-tune the retriever with LoRA using in-batch contrastive loss.

Usage (example, run on the RTX GPU machine):

    python -m vdocrag.retriever.train \
        --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
        --train_jsonl data/processed/retriever_train.jsonl \
        --image_root data/processed/images \
        --output_dir outputs/retriever-qwen2vl2b-lora \
        --per_device_train_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --load_in_4bit

Kept deliberately simple (plain PyTorch loop, not a Trainer subclass) so every
step is visible and easy to explain in a viva: batch -> encode query+doc ->
in-batch InfoNCE loss -> backward -> step.
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vdocrag.retriever.dataset import RetrieverContrastiveDataset, retriever_collate_fn
from vdocrag.retriever.modeling import VDocRetriever


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--query_max_len", type=int, default=256)
    p.add_argument("--doc_max_len", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--load_in_4bit", action="store_true",
                    help="Use 4-bit quantization (recommended for 8-16GB GPUs)")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    return p.parse_args()


def build_model(args):
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    if args.load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()

    model = VDocRetriever(base_model=base_model, temperature=args.temperature)
    return model, processor


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = build_model(args)
    model.train()

    dataset = RetrieverContrastiveDataset(args.train_jsonl, image_root=args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: retriever_collate_fn(
            b, processor, query_max_len=args.query_max_len, doc_max_len=args.doc_max_len
        ),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )

    global_step = 0
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch["query"] = {k: v.to(model.base_model.device) for k, v in batch["query"].items()}
            batch["document"] = {k: v.to(model.base_model.device) for k, v in batch["document"].items()}

            output = model(query=batch["query"], document=batch["document"])
            loss = output.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    pbar.set_postfix({"loss": output.loss.item(), "step": global_step})

                if global_step % args.save_steps == 0:
                    model.base_model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

    model.base_model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved final LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
