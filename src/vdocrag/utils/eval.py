"""
Evaluation metrics.

- Retrieval: Recall@k, nDCG@k (matches VDocRAG's reported metrics).
- QA: Exact Match (EM) and token-level F1 (standard QA metrics), plus a
  simple accuracy variant used by ChartQA/DocVQA-style benchmarks (relaxed
  match allowing minor formatting differences in numbers).
"""

import json
import re
import string
from collections import defaultdict
from typing import Dict, List


# ---------- Retrieval metrics ----------

def load_qrels(qrels_path: str) -> Dict[str, set]:
    """qrels format: qid \t docid \t relevance (1 = relevant)"""
    qrels = defaultdict(set)
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            qid, docid, rel = line.strip().split("\t")
            if int(rel) > 0:
                qrels[qid].add(docid)
    return qrels


def load_rankings(rank_path: str) -> Dict[str, List[str]]:
    rankings = defaultdict(list)
    with open(rank_path, encoding="utf-8") as f:
        for line in f:
            qid, docid, rank, score = line.strip().split("\t")
            rankings[qid].append(docid)
    return rankings


def recall_at_k(rankings: Dict[str, List[str]], qrels: Dict[str, set], k: int) -> float:
    hits, total = 0, 0
    for qid, relevant in qrels.items():
        retrieved = set(rankings.get(qid, [])[:k])
        if retrieved & relevant:
            hits += 1
        total += 1
    return hits / total if total else 0.0


def ndcg_at_k(rankings: Dict[str, List[str]], qrels: Dict[str, set], k: int) -> float:
    import math
    scores = []
    for qid, relevant in qrels.items():
        retrieved = rankings.get(qid, [])[:k]
        dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved) if d in relevant)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


# ---------- QA metrics ----------

def _normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def relaxed_accuracy(pred: str, gold: str, tolerance: float = 0.05) -> float:
    """ChartQA-style relaxed match: numeric answers within `tolerance` count as correct."""
    def try_float(x):
        try:
            return float(x.strip().rstrip("%").replace(",", ""))
        except ValueError:
            return None

    p_num, g_num = try_float(pred), try_float(gold)
    if p_num is not None and g_num is not None:
        if g_num == 0:
            return float(p_num == 0)
        return float(abs(p_num - g_num) / abs(g_num) <= tolerance)
    return exact_match(pred, gold)


def evaluate_qa(answers_json_path: str, gold_json_path: str) -> Dict[str, float]:
    preds = json.load(open(answers_json_path, encoding="utf-8"))
    golds = json.load(open(gold_json_path, encoding="utf-8"))

    em_scores, f1_scores, acc_scores = [], [], []
    for qid, gold_answer in golds.items():
        pred_answer = preds.get(qid, {}).get("answer", "")
        em_scores.append(exact_match(pred_answer, gold_answer))
        f1_scores.append(token_f1(pred_answer, gold_answer))
        acc_scores.append(relaxed_accuracy(pred_answer, gold_answer))

    n = len(golds)
    return {
        "exact_match": sum(em_scores) / n if n else 0.0,
        "f1": sum(f1_scores) / n if n else 0.0,
        "relaxed_accuracy": sum(acc_scores) / n if n else 0.0,
    }
