# CoT / Thinking Traces Research Notes

Research notes on how Chain-of-Thought (CoT) traces work in nanodistill, when they help (and when they don't), and broader considerations for encoder vs decoder architectures in classification tasks.

## Table of Contents

- [How CoT Traces Work in nanodistill](#how-cot-traces-work-in-nanodistill)
- [Does CoT Help with LoRA Fine-Tuning?](#does-cot-help-with-lora-fine-tuning)
- [Three Scenarios for CoT Usefulness](#three-scenarios-for-cot-usefulness)
- [Multi-Task CoT Distillation Approaches](#multi-task-cot-distillation-approaches)
- [Encoder vs Decoder for Classification](#encoder-vs-decoder-for-classification)
- [Other Considerations](#other-considerations)
- [Recommendation Framework](#recommendation-framework)

---

## How CoT Traces Work in nanodistill

nanodistill uses CoT traces across all three stages of its pipeline.

### Stage 1: Generation

`TeacherClient.synthesize_cot()` (`teacher/client.py:132`) sends each seed example to the teacher model (e.g., Claude) with a system prompt asking it to "show your step-by-step thinking process" (`teacher/prompts.py:14-21`). The response is parsed into a `ThinkingTrace` object (`teacher/schemas.py:8`) containing both the intermediate reasoning and the final answer. Results are saved to `traces_cot.jsonl`.

### Stage 2: Amplification

In the amplification pipeline (`amplifier/pipeline.py:45-60`), thinking traces feed into policy extraction. The prompt explicitly includes `"Reasoning: {trace.thinking}"` (`teacher/prompts.py:85`). The reasoning patterns from the teacher help the system understand *how* the task should be solved, which guides generation of synthetic training examples.

### Stage 3: Fine-Tuning

Each training example is formatted with the thinking content included (`distiller/trainer.py:162-184`). Two formats are supported:

- **Chat format:** Wrapped in `<thinking>...</thinking>` XML tags
- **Completion format:** Prefixed with `Thinking: ...`

The student model learns to generate reasoning steps as an intermediate step before producing the final answer.

### Why This Matters

1. **Teaching reasoning, not just answers.** Training the student to reproduce reasoning steps teaches generalizable problem-solving patterns rather than memorizing input-output mappings. This is especially important when seed data is small, which is nanodistill's primary use case.
2. **Better policy extraction.** Thinking traces improve amplification quality. The system understands *why* answers are correct, not just *what* they are.
3. **Improved generalization.** Research (e.g., "Distilling Step-by-Step," Hsieh et al., 2023) shows training smaller models on CoT traces from larger models significantly outperforms training on input-output pairs alone, often with much less data.

---

## Does CoT Help with LoRA Fine-Tuning?

**Yes, but more modestly than with full fine-tuning.**

The "Distilling Step-by-Step" paper used full fine-tuning, not LoRA. The gains may be smaller with LoRA for specific reasons:

### Why CoT helps *less* with LoRA

- LoRA only updates low-rank adapter matrices -- a tiny fraction of the model's parameters. It has limited capacity to learn entirely *new* reasoning patterns that the base model doesn't already have.
- Full fine-tuning can reshape the model's weights more deeply, absorbing complex reasoning chains more thoroughly.

### Why it still helps

- The base pretrained model (e.g., Llama, Mistral) already has reasoning ability from pretraining. LoRA doesn't need to teach reasoning from scratch -- it just needs to **steer** the model toward a specific task's reasoning style. The thinking traces give it a clearer signal for that steering.
- Without thinking traces, the model only sees `input -> output`. With them, it sees `input -> reasoning -> output`. Even with LoRA's limited capacity, this richer signal reduces ambiguity about what pattern to learn.
- It's more like "showing the path" to capabilities the model already has, rather than building new capabilities.

### Summary

CoT distillation with full fine-tuning can **teach** reasoning. CoT distillation with LoRA mostly **activates and formats** reasoning the model already has. Still helpful, but the ceiling is lower.

---

## Three Scenarios for CoT Usefulness

### Scenario 1: Encoder Classifier (BERT, RoBERTa, DeBERTa)

**CoT in training data is NOT used and NOT helpful.**

The training loop for a classifier is:

```
input_text -> Encoder -> [CLS] embedding -> Linear Head -> logits -> CrossEntropyLoss(logits, label)
```

The loss function is computed **only** on the logits vs. the ground-truth class label. There is no text generation objective. If the dataset has columns `[input, thinking, output_label]`, the `thinking` column is never part of the loss computation -- it's dead weight.

The only way CoT could enter the picture is concatenating it into the input text, but then you'd need CoT at inference time too. Training with CoT in the input but testing without it creates a **train/test distribution mismatch** that will hurt performance.

### Scenario 2: Decoder Model, No CoT at Inference

This is the most nuanced case.

**Naive approach (include CoT in training, strip at inference) -- likely hurts.** If you train on `Input -> Thinking -> Output` but at inference prompt for `Input -> Output`, you've created a distribution mismatch. The model learned that after seeing an input, the next tokens should be *reasoning*. Skipping that typically **degrades** performance compared to training directly on `Input -> Output`.

**Principled approaches that work:**

1. **Multi-task training** (Hsieh et al., 2023): Train with two objectives simultaneously -- one predicts the label, another generates the rationale. The rationale task acts as an auxiliary loss that shapes internal representations. Their 770M T5 model outperformed 540B PaLM.
2. **Implicit CoT via hidden-state distillation** (Deng et al., 2023): Train a teacher with explicit CoT, then distill into a student by aligning hidden states across layers. Reasoning happens "vertically" across layers rather than "horizontally" across tokens.
3. **Two-format training**: Mix both formats -- some examples with CoT, some without. At inference, use the direct format.

### Scenario 3: Decoder Model WITH CoT at Inference

This is what nanodistill currently does. The model generates reasoning then answers. Training and inference distributions match. Most beneficial.

### Summary Table

| Setup | CoT helps? | Why |
|-------|------------|-----|
| Encoder classifier | **No** | No generation objective; CoT never enters loss function |
| Decoder, no CoT at inference (naive) | **No, likely hurts** | Train/test distribution mismatch |
| Decoder, no CoT at inference (multi-task) | **Yes** | CoT as auxiliary training signal shapes representations |
| Decoder, no CoT at inference (hidden-state) | **Yes** | Reasoning internalized into layer representations |
| Decoder, CoT at inference | **Yes, most beneficial** | Full distribution match |

**Key insight:** CoT is not universally helpful. It depends on whether the architecture and training procedure have a mechanism to learn from it. Simply having CoT annotations in your dataset does nothing unless the loss function touches them.

---

## Multi-Task CoT Distillation Approaches

When you want the benefits of CoT during training but don't want to generate reasoning at inference time, multi-task training offers a principled solution.

### Core Concept

Instead of one training objective, use two:

| Task | Input | Target | Purpose |
|------|-------|--------|---------|
| **Primary** (label) | Question/input | Direct answer only | Used at inference |
| **Auxiliary** (rationale) | Question/input | CoT reasoning | Shapes internal representations during training |

The auxiliary task acts as a regularizer, forcing the model's hidden states to encode reasoning structure even though reasoning is never requested at inference.

### Approach A: Mixed-Format Training (Simplest)

Create the training dataset with both formats, distinguished by a task prefix:

```jsonl
{"text": "[DIRECT] Input: What is 23*47?\nOutput: 1081"}
{"text": "[REASON] Input: What is 23*47?\nReasoning: 23*47 = 23*40 + 23*7 = 920 + 161 = 1081\nOutput: 1081"}
```

At inference, always use the `[DIRECT]` prefix. The `[REASON]` examples improve the model's internal representations shared across both modes.

**Ratio matters.** Typical splits are 50/50 or 70% direct / 30% reasoning. Too much reasoning can bias the model toward generating extra tokens even in direct mode.

### Approach B: Two-Pass Loss (More Principled)

Compute two losses in a single training step:

```python
for batch in dataloader:
    inputs = batch["input"]
    labels = batch["output"]
    rationales = batch["thinking"]

    # Pass 1: Direct answer
    loss_label = compute_lm_loss(model, format_direct(inputs), labels)

    # Pass 2: Rationale generation
    loss_rationale = compute_lm_loss(model, format_reason(inputs), rationales)

    # Combined
    total_loss = loss_label + lambda_weight * loss_rationale
    total_loss.backward()
```

`lambda_weight` controls influence. Typical values: 0.1 to 0.5. Start with 0.3 and adjust based on validation.

### Approach C: Sequential Fine-Tuning (Pragmatic Middle Ground)

1. **Phase 1:** Fine-tune on `Input -> Thinking -> Output` (with CoT)
2. **Phase 2:** Fine-tune on `Input -> Output` (direct only, fewer steps)

Phase 1 shapes the model's representations. Phase 2 adapts the output format. Risk: Phase 2 can cause catastrophic forgetting of Phase 1's reasoning if you train too long. Use a lower learning rate and fewer steps in Phase 2.

### Why This Works

The auxiliary rationale task affects the model's shared hidden representations -- transformer layers that both tasks pass through. When the model learns to generate reasoning in the auxiliary task, those same layers develop richer intermediate representations (better attention patterns, more structured residual stream) that the direct-answer task also benefits from. This is the same mechanism behind why multi-task learning works in general: auxiliary tasks act as implicit regularization, preventing shallow shortcuts.

### With LoRA

Two options:

1. **Single LoRA adapter, mixed data (Approach A):** Simplest. Both tasks train the same adapter. Works well when tasks are related.
2. **Two separate LoRA adapters:** One for reasoning, another for direct answers. At inference, only load the direct-answer adapter. Cleaner separation but adapters don't share information, losing most multi-task benefit.

**Recommendation:** Single adapter with mixed data. The whole point is that the reasoning task improves the shared parameters.

### Practical Notes

- **Data efficiency:** Hsieh et al. showed this approach needs *less* training data than direct training alone -- their 770M model outperformed 540B PaLM with only 80% of available data.
- **Evaluation:** Always evaluate on the direct-answer format only.
- **Format leakage:** Watch for the model generating reasoning tokens even in direct mode. This means the ratio or prefix design needs adjustment.

---

## Encoder vs Decoder for Classification

### Architecture Recap

**Encoder (BERT, RoBERTa, DeBERTa):**

```
input -> Encoder (bidirectional attention) -> [CLS] embedding -> Linear -> logits
```

Every token sees every other token. Single forward pass produces class probabilities. Typically 110M-350M parameters.

**Decoder for classification (Llama, Qwen):**

```
Option A: input + prompt -> Decoder -> generates "positive"/"negative" (text)
Option B: input -> Decoder -> last-token embedding -> Linear -> logits
```

Each token only sees preceding tokens (causal attention). Much larger: 1B-70B+ parameters.

### Who Wins?

For standard classification tasks (sentiment, topic, NLI, intent), **fine-tuned encoder models consistently match or beat decoder models that are 10-100x larger.**

| Model | Type | Params | Typical Performance |
|-------|------|--------|---------------------|
| DeBERTa-v3-large | Encoder | 304M | State-of-art on SuperGLUE (~91) |
| RoBERTa-large | Encoder | 355M | Extremely strong on most benchmarks |
| Llama-2-7B (fine-tuned) | Decoder | 7B | Competitive but rarely better |
| GPT-3 (few-shot) | Decoder | 175B | Often worse than fine-tuned BERT |

### Why Encoders Win on Pure Classification

1. **Bidirectional attention is inherently better for understanding.** Classification is a comprehension task, not a generation task. An encoder processes "The movie was not bad" with full bidirectional context. A decoder processes "not" before seeing "bad," then "bad" without seeing what comes after.
2. **Pre-training alignment.** Masked Language Modeling forces the model to understand context from both directions -- directly aligned with classification. Next-token prediction optimizes for generation, a different skill.
3. **Parameter efficiency.** An encoder dedicates 100% of its parameters to understanding. A decoder splits capacity between understanding and generation.

### When Decoders Win

| Scenario | Why |
|----------|-----|
| Very few labels (<500 examples) | In-context learning with large LLM, no fine-tuning needed |
| Classification requires complex reasoning | CoT can bridge multi-step inference that attention alone can't |
| Open-ended / dynamic label space | Encoder requires fixed classes at training; decoder can classify into new categories |
| Need explanations with predictions | Decoder can generate rationale; encoder only gives logits |
| Multi-task deployment | One decoder handles classification + summarization + QA; encoder needs one head per task |

### CoT's Role for Each Architecture

**Encoder:** CoT in training data has no architectural mechanism to be used. Options are limited to indirect benefits:

1. **Data quality improvement:** Use CoT to verify labels, filter noisy examples, weight examples by reasoning quality. The CoT never enters the model -- it's a data curation tool.
2. **Input augmentation:** Extract key phrases or structured features from reasoning and add them to input text. E.g., instead of just `"The battery lasts 2 hours"`, feed `"The battery lasts 2 hours [ASPECT: battery_life] [SENTIMENT_CUE: short_duration]"`. This is manual feature engineering, not CoT training.

There is no option where encoders learn from CoT in the output.

**Decoder:** CoT is usable via multi-task training (see previous section). But consider whether the reasoning actually helps. For simple sentiment classification, CoT adds little. For complex multi-hop reasoning ("Is this contract clause legally valid given clauses 3 and 7?"), CoT adds a lot.

---

## Other Considerations

### Data Size

| Data Amount | Recommendation |
|-------------|---------------|
| <100 labeled examples | Large decoder with in-context learning (no fine-tuning) |
| 100-1,000 | Few-shot fine-tuning of decoder, OR encoder with data augmentation |
| 1,000-100K | Fine-tuned encoder almost always wins |
| 100K+ | Both work; encoder is far more cost-efficient |

### Inference Latency and Cost

Often the deciding factor in production:

- **Encoder:** ~5-10ms per classification (single forward pass, small model)
- **Decoder (embedding mode):** ~50-200ms (larger model, single pass)
- **Decoder (generative mode):** ~500-5000ms (autoregressive token generation)
- **Decoder with CoT:** ~2-10 seconds (generates many reasoning tokens before answer)

At millions of classifications per day, the encoder is orders of magnitude cheaper.

### Calibration

Encoder models produce **better-calibrated probabilities**. When the encoder says 0.85 confidence, that's more reliable than a decoder's 0.85. Matters for:

- Threshold-based decisions ("flag if confidence < 0.7")
- Cascading systems ("if uncertain, escalate to human review")
- Risk-sensitive domains (medical, financial)

### Label Noise

If labels might be wrong, CoT annotations are valuable as a **data cleaning tool** regardless of architecture:

```
Label: positive
Reasoning: "The reviewer mentions 'not bad' which is actually neutral/lukewarm..."
```

Use reasoning to identify and fix mislabeled examples *before* training.

### Class Imbalance

- **Encoder:** Use weighted loss, oversampling, focal loss -- well-studied techniques.
- **Decoder:** Can use CoT to generate synthetic examples for minority classes. The reasoning helps generate more diverse and realistic synthetic data.

### Interpretability

- **Encoder only:** You get logits. For interpretability, need post-hoc methods (attention visualization, LIME, SHAP) which are approximate.
- **Decoder with CoT:** You get human-readable reasoning. But beware -- the reasoning may be a post-hoc rationalization rather than the model's actual decision process (known issue with LLM CoT).

---

## Recommendation Framework

### Decision Tree

```
Q1: Do you need explanations alongside classifications?
  YES -> Decoder with multi-task CoT training
  NO  -> Q2

Q2: How much labeled training data?
  <500 examples -> Large decoder with in-context learning
  500+ examples -> Q3

Q3: Does classification require multi-step reasoning?
  YES (legal analysis, medical diagnosis) -> Decoder with CoT
  NO  (sentiment, topic, intent)          -> Q4

Q4: What are your latency/cost constraints?
  Tight (production, high volume)  -> Encoder (DeBERTa-v3)
  Loose (batch processing, offline) -> Either works, encoder still more efficient

Default: Fine-tuned DeBERTa-v3-large
```

### Hybrid Approach (Often the Best Answer)

Use CoT and decoder models for **data preparation**, then train an encoder for **production inference**:

1. Use a large teacher model to generate CoT for training examples
2. Use CoT to **verify and clean labels** (flag inconsistencies between reasoning and labels)
3. Use CoT to **generate synthetic training data** for underrepresented classes
4. Train a small, fast encoder on the cleaned/augmented dataset
5. Deploy the encoder in production

This gets the benefits of CoT (better data quality, more training data) with the efficiency of an encoder (fast, cheap, well-calibrated).

---

## References

- Hsieh et al., "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes," ACL 2023
- Deng et al., "Implicit Chain of Thought Reasoning via Knowledge Distillation," 2023
