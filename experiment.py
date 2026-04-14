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
     "Penicillin was discovered by Alexander Fleming in 1928.",
     "Penicillin was discovered by Alexander Fleming.", 5, False),

    ("What is the chemical symbol for gold?",
     "The chemical symbol for gold is Au, from the Latin word aurum.",
     "The chemical symbol for gold is Au.", 5, False),

    ("What is the boiling point of water?",
     "The boiling point of water at sea level is 100 degrees Celsius.",
     "Water boils at 100 degrees Celsius at sea level.", 5, False),

    ("When did the Berlin Wall fall?",
     "The Berlin Wall fell on November 9, 1989.",
     "The Berlin Wall fell in 1989.", 5, False),

    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519.",
     "Leonardo da Vinci painted the Mona Lisa.", 5, False),

    ("What is the largest planet?",
     "Jupiter is the largest planet in the solar system, with mass twice all others.",
     "Jupiter is the largest planet in the solar system.", 5, False),

    ("Who wrote Romeo and Juliet?",
     "Romeo and Juliet was written by William Shakespeare in the 1590s.",
     "Shakespeare wrote Romeo and Juliet.", 5, False),

    ("What does DNA stand for?",
     "DNA stands for deoxyribonucleic acid, the molecule carrying genetic instructions.",
     "DNA stands for deoxyribonucleic acid.", 5, False),

    # ── PARTIAL answers (human score 3) — vague or incomplete ───────────────
    ("Who wrote Hamlet?",
     "Hamlet was written by William Shakespeare, probably between 1599 and 1601.",
     "It was written by a famous English playwright.", 3, False),

    ("What is the capital of France?",
     "The capital of France is Paris, located on the River Seine.",
     "France has a capital city in Europe.", 3, False),

    ("Who discovered penicillin?",
     "Penicillin was discovered by Alexander Fleming in 1928.",
     "An English scientist discovered it in the early 20th century.", 3, False),

    ("What is the chemical symbol for gold?",
     "The chemical symbol for gold is Au, from the Latin word aurum.",
     "Gold has a two-letter chemical symbol.", 3, False),

    ("What is the boiling point of water?",
     "The boiling point of water at sea level is 100 degrees Celsius.",
     "Water boils at a temperature above 90 degrees.", 3, False),

    ("When did the Berlin Wall fall?",
     "The Berlin Wall fell on November 9, 1989.",
     "The Berlin Wall fell sometime in the late 1980s.", 3, False),

    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519.",
     "An Italian Renaissance artist painted it.", 3, False),

    ("What is the largest planet?",
     "Jupiter is the largest planet in the solar system, with mass twice all others.",
     "One of the gas giants is the largest planet.", 3, False),

    ("Who wrote Romeo and Juliet?",
     "Romeo and Juliet was written by William Shakespeare in the 1590s.",
     "The same person who wrote Hamlet also wrote this play.", 3, False),

    ("What does DNA stand for?",
     "DNA stands for deoxyribonucleic acid, the molecule carrying genetic instructions.",
     "DNA is an acronym for a type of biological molecule.", 3, False),

    # ── HALLUCINATED answers (human score 1) — plausible but factually wrong ─
    ("Who wrote Hamlet?",
     "Hamlet was written by William Shakespeare, probably between 1599 and 1601.",
     "Hamlet was written by Christopher Marlowe in 1592.", 1, True),

    ("What is the capital of France?",
     "The capital of France is Paris, located on the River Seine.",
     "The capital of France is Lyon, the country's second-largest city.", 1, True),

    ("Who discovered penicillin?",
     "Penicillin was discovered by Alexander Fleming in 1928.",
     "Penicillin was discovered by Louis Pasteur in the 1880s.", 1, True),

    ("What is the chemical symbol for gold?",
     "The chemical symbol for gold is Au, from the Latin word aurum.",
     "The chemical symbol for gold is Gd, standing for goldium.", 1, True),

    ("What is the boiling point of water?",
     "The boiling point of water at sea level is 100 degrees Celsius.",
     "Water boils at 212 degrees Celsius at sea level.", 1, True),

