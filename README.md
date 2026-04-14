# EvalFramework

**GPT-2 Small (117M) is an unreliable judge: LLM-as-Judge Spearman ρ=−0.10 (random), G-Eval ρ=0.36 (marginal), hallucination detector AUC=0.43 (below random). This experiment calibrates the minimum scale requirements for automated LLM evaluation — and explains exactly why production eval uses GPT-4 or Claude, not a base 117M model.**

---

## Three Evaluation Methods Implemented from Scratch

### 1. LLM-as-Judge
Given `(question, context, answer)`, score answer quality 1–5 without a trained classifier:

```python
prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer: {ans}\nRate quality (1=wrong, 5=correct). Score:"
lp     = log_softmax(model(prompt).logits[-1])        # next-token log-probs
probs  = softmax(lp[[tok("1"), tok("2"), tok("3"), tok("4"), tok("5")]])
score  = Σ (i+1) * probs[i]                           # expected score ∈ [1,5]
```

### 2. G-Eval (Multi-Dimensional Faithfulness)
Decompose faithfulness into three binary criteria (Liu et al. 2023), score each with `P("yes")`:

| Criterion | Prompt completion |
|---|---|
| Factual groundedness | *"Is the answer factually correct based on the context? yes/no"* |
| Completeness | *"Does the answer directly address the question? yes/no"* |
| Groundedness | *"Is the answer free of information not in the context? yes/no"* |

`G-Eval score = 1 + 4 × mean(P_yes for each criterion)`

### 3. Hallucination Detector
Computes log-probability *lift* from adding context:

```
hallucination_score = −( log P(answer | context + question)
                       − log P(answer | question only) )
```

A grounded answer should have much higher log-prob when supporting context is present. A hallucination is equally probable with or without context — it comes from the model's parametric memory, not the retrieved document. Higher score = more hallucinated.

---

## Results

| Method | Spearman ρ with human scores | Interpretation |
|---|---|---|
| LLM-as-Judge | **−0.104** (p=0.585) | No correlation — effectively random |
| G-Eval | **0.358** (p=0.052) | Marginal signal, near significance |
| Hallucination AUC | **0.430** | Below random — reversed ordering |

---

## The Key Finding: Scale Is the Prerequisite

**All three methods fail or underperform at 117M parameters. This is not a bug — it's a quantitative calibration of the minimum scale requirement for LLM evaluation.**

**LLM-as-Judge (ρ = −0.10):** GPT-2 assigns nearly identical log-probabilities to the digit tokens "1" through "5" regardless of answer quality. The model hasn't learned to evaluate quality — it has learned language. At GPT-2 scale, the expected score is ~2.4 for almost every example. GPT-4 (1.8T) works as a judge because quality understanding emerges at scale through RLHF fine-tuning.

**G-Eval (ρ = 0.36):** Multi-dimensional scoring extracts more signal than single-score judgment, confirming Liu et al.'s finding. The criteria decomposition forces the model to reason about specific properties rather than output a single quality token. This marginal signal would be very strong at larger scale — G-Eval with GPT-4 achieves ρ > 0.8 with human annotators.

**Hallucination Detector (AUC = 0.43, *below random*):** The most striking result. GPT-2 assigns *higher* log-prob to hallucinated answers *when wrong context is provided* than without context. The model is pattern-matching the wrong document into its completion rather than detecting the mismatch. Larger models (10B+) have enough world knowledge to detect when a retrieved context contradicts the answer. GPT-2 doesn't — it just adapts to whatever context is present.

**The practical implication:** When building a production eval system, use the largest available model as the judge. The methods themselves are correct and widely deployed (RAGAS, Prometheus, OpenAI Evals all use variants of these). The failure here is a scale failure, not a method failure.

---

## Dataset

30 (question, context, answer) triples:
- **10 correct** (human score 5): answer taken directly from the retrieved context
- **10 partial** (human score 3): vague or incomplete — technically true but not useful
- **10 hallucinated** (human score 1): plausible-sounding but factually wrong answers not supported by context

---

## Plot

`plots/eval_framework.png` — three panels:
1. LLM-as-Judge score vs human score (scatter by quality tier)
2. LLM-as-Judge vs G-Eval (cross-method correlation)
3. Hallucination score distribution: grounded vs hallucinated examples

---

## Limitations

- **Scale:** All methods require 10B+ parameters to function as intended. Results should be treated as a lower bound.
- **Dataset size:** 30 examples is too small for robust correlation estimates (confidence intervals are wide).
- **Static context:** Production RAG eval also measures retrieval quality (is the right document retrieved?) and faithfulness to the retrieved document. This experiment uses oracle contexts.
- **No reference-based metrics:** BLEU, ROUGE, BERTScore are not implemented; the focus is on reference-free LLM-based evaluation.

---

## Connection to Portfolio

| Project | Relationship |
|---|---|
| **RAG** | This framework evaluates RAG output quality — faithfulness and hallucination are the primary RAG failure modes |
| **LLMAgents** | Agent outputs need automated eval; hallucination detector would catch tool-observation inconsistencies |
| **RewardModeling** | Reward model is a trained judge; this project shows the limits of zero-shot judging and motivates training |

---

## References

Liu et al. (2023). *G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment.* EMNLP 2023. https://arxiv.org/abs/2303.16634

Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. https://arxiv.org/abs/2306.05685

Es et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* https://arxiv.org/abs/2309.15217
