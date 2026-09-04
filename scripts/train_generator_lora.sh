#!/bin/bash
# Phase 2: our own from-scratch LoRA SFT of the generator on Qwen2-VL-2B.
# Run this on the RTX GPU machine.
set -e

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
