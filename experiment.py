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

    ("When did the Berlin Wall fall?",
     "The Berlin Wall fell on November 9, 1989.",
     "The Berlin Wall fell in November 1991 during the Soviet collapse.", 1, True),

    ("Who painted the Mona Lisa?",
     "The Mona Lisa was painted by Leonardo da Vinci, likely between 1503 and 1519.",
     "The Mona Lisa was painted by Michelangelo during the Renaissance.", 1, True),

    ("What is the largest planet?",
     "Jupiter is the largest planet in the solar system, with mass twice all others.",
     "Saturn is the largest planet because of its enormous ring system.", 1, True),

    ("Who wrote Romeo and Juliet?",
     "Romeo and Juliet was written by William Shakespeare in the 1590s.",
     "Romeo and Juliet was written by John Keats as a narrative poem.", 1, True),

    ("What does DNA stand for?",
     "DNA stands for deoxyribonucleic acid, the molecule carrying genetic instructions.",
     "DNA stands for Digital Nucleotide Assembly, used in genetic sequencing.", 1, True),
]

human_scores = np.array([d[3] for d in DATA])
is_hallucination = np.array([d[4] for d in DATA])

# ── Load model ────────────────────────────────────────────────────────────────
print("=" * 58)
print("EvalFramework — LLM-as-Judge · G-Eval · Hallucination Detect")
print("=" * 58)
print(f"\n  Examples: {len(DATA)} ({sum(1 for d in DATA if d[3]==5)} correct, "
      f"{sum(1 for d in DATA if d[3]==3)} partial, "
      f"{sum(1 for d in DATA if d[3]==1)} hallucinated)")
print(f"  Device  : {DEVICE}")

tokeniser = GPT2Tokenizer.from_pretrained("gpt2")
tokeniser.pad_token = tokeniser.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
model.eval()

# Digit token IDs for scoring 1-5
DIGIT_IDS = [tokeniser(f" {i}", add_special_tokens=False)["input_ids"][0] for i in range(1, 6)]
# Yes/No token IDs
YES_ID = tokeniser(" yes", add_special_tokens=False)["input_ids"][0]
NO_ID  = tokeniser(" no",  add_special_tokens=False)["input_ids"][0]

@torch.no_grad()
def next_token_logprobs(prompt: str) -> torch.Tensor:
    enc = tokeniser(prompt, return_tensors="pt", truncation=True,
                    max_length=256).to(DEVICE)
    logits = model(**enc).logits[0, -1, :]
    return torch.log_softmax(logits, dim=-1)

@torch.no_grad()
def answer_logprob(question: str, context: str, answer: str, use_context: bool) -> float:
    if use_context:
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
    else:
        prompt = f"Question: {question}\nAnswer: {answer}"
    ctx_enc  = tokeniser(prompt[:-len(answer)], return_tensors="pt",
                         truncation=True, max_length=200).to(DEVICE)
    full_enc = tokeniser(prompt, return_tensors="pt",
                         truncation=True, max_length=220).to(DEVICE)
    full_logits = model(**full_enc).logits[0]
    ctx_len = ctx_enc["input_ids"].shape[1]
    ans_ids = full_enc["input_ids"][0, ctx_len:]
    lp = 0.0
    for i, tok in enumerate(ans_ids):
        pos = ctx_len + i - 1
        if pos < full_logits.shape[0]:
            lp += torch.log_softmax(full_logits[pos], dim=-1)[tok].item()
    return lp

# ── Method 1: LLM-as-Judge ────────────────────────────────────────────────────
def llm_judge(question, context, answer) -> float:
    prompt = (f"Context: {context}\nQuestion: {question}\nAnswer: {answer}\n"
              f"Rate the answer quality (1=wrong/bad, 3=partial, 5=correct/complete).\n"
              f"Score:")
    lp = next_token_logprobs(prompt)
    digit_lp = lp[DIGIT_IDS]
    probs    = (digit_lp - digit_lp.logsumexp(0)).exp()
    return float(sum((i+1) * probs[i].item() for i in range(5)))

# ── Method 2: G-Eval (multi-dimensional faithfulness) ────────────────────────
def geval_score(question, context, answer) -> float:
    scores = []
    criteria = [
        ("Is the answer factually correct based on the context above? Answer yes or no:",        YES_ID, NO_ID),
        ("Does the answer directly address the question? Answer yes or no:",                     YES_ID, NO_ID),
        ("Is the answer free of information not present in the context? Answer yes or no:",      YES_ID, NO_ID),
    ]
    for crit_text, yes_id, no_id in criteria:
        prompt = (f"Context: {context}\nQuestion: {question}\nAnswer: {answer}\n{crit_text}")
        lp = next_token_logprobs(prompt)
        yes_lp = lp[yes_id].item()
        no_lp  = lp[no_id].item()
        p_yes  = float(torch.tensor([yes_lp, no_lp]).softmax(0)[0])
        scores.append(p_yes)
    # Map [0,1] → [1,5]
    return 1 + 4 * float(np.mean(scores))

# ── Method 3: Hallucination Detector ─────────────────────────────────────────
def hallucination_score(question, context, answer) -> float:
    lp_with    = answer_logprob(question, context, answer, use_context=True)
    lp_without = answer_logprob(question, context, answer, use_context=False)
    # Negative Δ: higher score = more hallucinated
    return -(lp_with - lp_without)

# ── Run evaluations ───────────────────────────────────────────────────────────
print("\nRunning 3 eval methods × 30 examples...")
judge_scores, geval_scores, hall_scores = [], [], []

for i, (q, ctx, ans, hs, is_hall) in enumerate(DATA):
    j  = llm_judge(q, ctx, ans)
    g  = geval_score(q, ctx, ans)
    h  = hallucination_score(q, ctx, ans)
    judge_scores.append(j); geval_scores.append(g); hall_scores.append(h)
    if (i+1) % 10 == 0:
        print(f"  [{i+1}/30] judge={j:.2f} g-eval={g:.2f} hallucination_score={h:.2f} "
              f"(human={hs}, hallucinated={'Y' if is_hall else 'N'})")

judge_scores = np.array(judge_scores)
geval_scores = np.array(geval_scores)
hall_scores  = np.array(hall_scores)

# Spearman correlations
rho_judge, p_judge = spearmanr(human_scores, judge_scores)
rho_geval, p_geval = spearmanr(human_scores, geval_scores)

# Hallucination AUC (manual, binary labels: is_hallucination)
from sklearn.metrics import roc_auc_score, precision_recall_curve
hall_labels = is_hallucination.astype(int)
hall_auc = roc_auc_score(hall_labels, hall_scores) if hall_scores.std() > 0 else 0.5
prec, rec, thr = precision_recall_curve(hall_labels, hall_scores)
# Best F1
f1 = 2*prec*rec/(prec+rec+1e-9)
best_idx = f1.argmax()
best_prec, best_rec, best_f1 = prec[best_idx], rec[best_idx], f1[best_idx]

print(f"\n  LLM-as-Judge   Spearman ρ = {rho_judge:.3f}  (p={p_judge:.3f})")
print(f"  G-Eval         Spearman ρ = {rho_geval:.3f}  (p={p_geval:.3f})")
print(f"  Halluc. Detect AUC = {hall_auc:.3f}  best-F1 = {best_f1:.3f}  "
      f"(P={best_prec:.2f} R={best_rec:.2f})")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
SCORE_COLOR = {5: "#16a34a", 3: "#f59e0b", 1: "#dc2626"}
colors_by_score = [SCORE_COLOR[d[3]] for d in DATA]
labels_by_score = {5: "Correct (human=5)", 3: "Partial (human=3)", 1: "Hallucinated (human=1)"}

ax = axes[0]
for score, color, label in [(5,"#16a34a","Correct (5)"), (3,"#f59e0b","Partial (3)"), (1,"#dc2626","Hallucinated (1)")]:
    idx = [i for i,d in enumerate(DATA) if d[3]==score]
    ax.scatter([human_scores[i] + np.random.uniform(-0.1,0.1) for i in idx],
               [judge_scores[i] for i in idx], s=60, color=color, label=label, alpha=0.85)
ax.set_xlabel("Human score"); ax.set_ylabel("LLM-as-Judge score")
ax.set_title(f"LLM-as-Judge vs Human\nSpearman ρ={rho_judge:.3f}", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for score, color in [(5,"#16a34a"), (3,"#f59e0b"), (1,"#dc2626")]:
    idx = [i for i,d in enumerate(DATA) if d[3]==score]
    ax.scatter([judge_scores[i] for i in idx],
               [geval_scores[i] for i in idx], s=60, color=color, alpha=0.85)
ax.set_xlabel("LLM-as-Judge"); ax.set_ylabel("G-Eval (faithfulness)")
ax.set_title(f"LLM-as-Judge vs G-Eval\nG-Eval Spearman ρ={rho_geval:.3f}", fontsize=10)
ax.grid(alpha=0.3)

ax = axes[2]
grounded_scores = hall_scores[~is_hallucination]
halluc_scores   = hall_scores[is_hallucination]
ax.hist(grounded_scores, bins=12, alpha=0.7, color="#16a34a", label=f"Grounded (n={len(grounded_scores)})")
ax.hist(halluc_scores,   bins=12, alpha=0.7, color="#dc2626", label=f"Hallucinated (n={len(halluc_scores)})")
ax.set_xlabel("Hallucination score (higher = more hallucinated)")
ax.set_ylabel("Count")
ax.set_title(f"Hallucination Detector\nAUC={hall_auc:.3f}  best-F1={best_f1:.3f}", fontsize=10)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle(f"LLM Evaluation Framework — GPT-2 Small | {DEVICE} | No training\n"
             f"LLM-as-Judge ρ={rho_judge:.3f} · G-Eval ρ={rho_geval:.3f} · Halluc AUC={hall_auc:.3f}",
             fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "eval_framework.png"), dpi=150, bbox_inches="tight")
plt.close(fig); print("\n  Saved → eval_framework.png")

print()
print("=" * 58)
print("EXPERIMENT COMPLETE")
print("=" * 58)
print(f"  LLM-as-Judge   Spearman ρ : {rho_judge:.3f}")
print(f"  G-Eval         Spearman ρ : {rho_geval:.3f}")
print(f"  Halluc. Detect AUC        : {hall_auc:.3f}  F1={best_f1:.3f}")
print(f"\nPlots → {PLOTS_DIR}/")
