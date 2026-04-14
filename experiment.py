"""
EvalFramework — LLM Evaluation from Scratch
============================================
Implements three production-grade evaluation methods without any eval
library (no deepeval, no RAGAS, no promptfoo):

  1. LLM-as-Judge
     Given (question, answer), ask GPT-2 to score quality 1–5 by
     computing log P("1"|context) … log P("5"|context), normalising
     to a probability distribution, and returning the expected score.
     Correlate with human scores using Spearman ρ.

  2. G-Eval (faithfulness scoring)
     Decompose faithfulness into three criteria:
       (a) Factual groundedness — is the claim in the retrieved context?
       (b) Completeness        — does it answer what was asked?
       (c) Conciseness         — is it free of irrelevant padding?
     Score each criterion independently (log-prob of "yes"/"no"),
     then aggregate. Multi-dimensional scoring is better calibrated
     than single-score judges (G-Eval, Liu et al. 2023).

  3. Hallucination Detector
     Compute log P(answer | context + question) vs
              log P(answer | question only).
     A grounded answer should have much higher log-prob when the
     supporting context is present. A hallucinated answer is equally
     probable with or without context (it comes from memorised
     parametric knowledge, not the retrieved document).
     Hallucination score = −(Δ log-prob) — higher means more hallucinated.

Dataset: 30 (question, context, answer, human_score) examples
  10 correct answers    (human score 5) — answer directly from context
  10 partial answers    (human score 3) — vague / incomplete
  10 hallucinated       (human score 1, hallucination=True) — plausible but wrong

M4 design: MPS, all inference, no training. Each eval method runs in
a single batched forward pass per example. ~30 seconds total.

Industry context:
  Evaluation is the hardest unsolved problem in production LLM systems.
  This framework implements the methods used in:
  - GPT-4 technical report (LLM-as-judge for model comparison)
  - RAGAS (faithfulness, answer relevancy metrics for RAG)
  - G-Eval (multi-dimensional NLG evaluation, Liu et al. 2023)
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEED      = 42; torch.manual_seed(SEED); np.random.seed(SEED)
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Dataset: 30 examples with human scores ───────────────────────────────────
# Each: (question, context_fact, answer, human_score[1-5], is_hallucination)
DATA = [
    # ── CORRECT answers (human score 5) — answer directly from context ──────
    ("Who wrote Hamlet?",
     "Hamlet was written by William Shakespeare, probably between 1599 and 1601.",
     "Shakespeare wrote Hamlet.", 5, False),

    ("What is the capital of France?",
     "The capital of France is Paris, located on the River Seine.",
     "The capital of France is Paris.", 5, False),

    ("Who discovered penicillin?",
