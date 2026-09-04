#!/bin/bash
# Phase 1: reproduce VDocRAG using NTT's released pretrained checkpoints.
# Inference-only -- runs fine on CPU (slow) or any GPU, no training needed.
set -e

QUERY_DATASET=chartqa
CORPUS_DATASET=chartqa
EMBEDDING_OUTPUT_DIR=outputs/baseline_embeddings
mkdir -p $EMBEDDING_OUTPUT_DIR

echo "== Encoding queries with NTT's pretrained VDocRetriever =="
python -m vdocrag.retriever.encode \
  --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
  --lora_name_or_path NTT-hil-insight/VDocRetriever-Phi3-vision \
  --input_jsonl data/processed/test_queries.jsonl \
  --mode query \
  --output_path $EMBEDDING_OUTPUT_DIR/query.pkl

echo "== Encoding corpus with NTT's pretrained VDocRetriever =="
python -m vdocrag.retriever.encode \
  --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
  --lora_name_or_path NTT-hil-insight/VDocRetriever-Phi3-vision \
  --input_jsonl data/processed/corpus.jsonl \
  --image_root data/processed/images \
  --mode document \
  --output_path $EMBEDDING_OUTPUT_DIR/corpus.pkl

echo "== Retrieving top-k =="
python -m vdocrag.retriever.search \
  --query_embeddings $EMBEDDING_OUTPUT_DIR/query.pkl \
  --corpus_embeddings $EMBEDDING_OUTPUT_DIR/corpus.pkl \
  --top_k 10 \
  --output_path $EMBEDDING_OUTPUT_DIR/rank.$QUERY_DATASET.$CORPUS_DATASET.txt

echo "== Generating answers with NTT's pretrained VDocGenerator =="
python -m vdocrag.generator.generate \
  --model_name_or_path microsoft/Phi-3-vision-128k-instruct \
  --lora_name_or_path NTT-hil-insight/VDocGenerator-Phi3-vision \
  --queries_jsonl data/processed/test_queries.jsonl \
  --corpus_jsonl data/processed/corpus.jsonl \
  --image_root data/processed/images \
  --rank_file $EMBEDDING_OUTPUT_DIR/rank.$QUERY_DATASET.$CORPUS_DATASET.txt \
  --top_k 3 \
  --output_path outputs/answers/baseline_answers.json

echo "Done. See outputs/answers/baseline_answers.json"
