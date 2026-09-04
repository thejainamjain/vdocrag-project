"""
LoRA SFT for the generator: (question + retrieved page images) -> answer.

Usage:
    python -m vdocrag.generator.train \
        --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
        --train_jsonl data/processed/generator_train.jsonl \
        --image_root data/processed/images \
        --output_dir outputs/generator-qwen2vl2b-lora \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 1e-4 \
        --num_train_epochs 1 \
        --load_in_4bit
"""

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from vdocrag.generator.dataset import GeneratorSFTDataset, generator_collate_fn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, default="")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=16)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    return p.parse_args()


def build_model(args):
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4",
        )

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, quantization_config=quant_config,
        torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
    )

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, processor = build_model(args)
    model.train()

    dataset = GeneratorSFTDataset(args.train_jsonl, image_root=args.image_root)
    loader = DataLoader(
        dataset, batch_size=args.per_device_train_batch_size, shuffle=True,
        collate_fn=lambda b: generator_collate_fn(b, processor, max_len=args.max_len),
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)

    global_step = 0
    for epoch in range(args.num_train_epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    pbar.set_postfix({"loss": outputs.loss.item(), "step": global_step})

                if global_step % args.save_steps == 0:
                    model.save_pretrained(os.path.join(args.output_dir, f"checkpoint-{global_step}"))

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved final generator LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
