# Qwen3 fine-tuning for Yoruba POS tagging

This repository is a compact set of experiments for adapting **Qwen3-0.6B** to part-of-speech tagging in **Yorùbá**. I use the same task formulation and evaluation pipeline across several settings so the main differences come from the adaptation method rather than from changes in the prompt or test procedure.

The experiments cover zero-shot and few-shot prompting, full supervised fine-tuning, LoRA, QLoRA, and a manual PyTorch implementation of the LoRA training loop.

## What is being predicted?

The data comes from **MasakhaPOS** through the Hugging Face `datasets` library:

```python
load_dataset("masakhane/masakhapos", "yor")
```

Each example contains a tokenized sentence and a sequence of Universal POS labels:

```text
tokens -> ["...", "...", "..."]
upos   -> [tag_id, tag_id, tag_id]
```

The scripts read the tag names directly from the dataset metadata and turn the task into an instruction-following problem. Given the token list, Qwen is asked to return one UPOS tag per token as a JSON list.

A training example therefore has the form:

```text
system:    task definition and allowed POS tags
user:      token list
assistant: JSON list of POS tags
```

The output is parsed strictly. A prediction is considered invalid when it is not valid JSON, is not a list, has the wrong number of labels, or contains a tag outside the dataset label set.

## Experiments

| Script | Training | What changes |
| --- | --- | --- |
| `qwen_base_zero_shot.py` | No | Base Qwen3 model receives only the task instruction |
| `qwen_few_shot.py` | No | Adds 5 labeled examples from the training split to the prompt |
| `qwen_full_finetune.py` | Yes | Updates all model parameters |
| `qwen_lora.py` | Yes | Trains low-rank adapters with TRL + PEFT |
| `qwen_lora_pytorch.py` | Yes | Uses the same LoRA idea but implements the training loop explicitly in PyTorch |
| `qwen_qlora.py` | Yes | Loads the base model in 4-bit NF4 and trains LoRA adapters |

The manual PyTorch script is not a new adaptation algorithm. It is an implementation variant of LoRA. I keep it because it makes the training mechanics visible: `DataLoader`, batching, assistant-only label masking, forward pass, backward pass, gradient accumulation, gradient clipping, optimizer updates, learning-rate scheduling, validation, and CUDA memory measurement are all handled explicitly.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── .gitignore
└── experiments
    ├── qwen_base_zero_shot.py
    ├── qwen_few_shot.py
    ├── qwen_full_finetune.py
    ├── qwen_lora.py
    ├── qwen_lora_pytorch.py
    └── qwen_qlora.py
```

The experiment scripts are deliberately self-contained. There is some repeated preprocessing and evaluation code, but this makes each file runnable and readable on its own.

## Setup

A CUDA-capable GPU is required for these experiments.

```bash
git clone https://github.com/faridbagheri/Fine-Tuning.git
cd Fine-Tuning

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For QLoRA, `bitsandbytes` must be supported by the CUDA/PyTorch environment.

## Running the experiments

Zero-shot:

```bash
python experiments/qwen_base_zero_shot.py
```

Few-shot:

```bash
python experiments/qwen_few_shot.py
```

Full fine-tuning:

```bash
python experiments/qwen_full_finetune.py
```

LoRA with TRL/PEFT:

```bash
python experiments/qwen_lora.py
```

LoRA with an explicit PyTorch training loop:

```bash
python experiments/qwen_lora_pytorch.py
```

QLoRA:

```bash
python experiments/qwen_qlora.py
```

Each script writes a CSV summary after evaluation. Fine-tuning scripts also save the trained model or adapter locally. Generated CSV files and model directories are ignored by Git because they are run artifacts rather than source code.

## Shared experimental setup

The scripts use the following common choices wherever they apply:

- model: `Qwen/Qwen3-0.6B`
- dataset: `masakhane/masakhapos`
- language: `yor`
- random seed: `42`
- maximum training sequence length: `512`
- train batch size: `1`
- evaluation batch size: `1`
- gradient accumulation: `8`
- training epochs: `3`
- optimizer: AdamW
- scheduler: cosine
- warmup: `10` optimizer steps
- greedy generation (`do_sample=False`)
- accuracy, macro F1, and invalid-output rate for test evaluation

Full fine-tuning uses a learning rate of `2e-5`. LoRA and QLoRA use `2e-4`, which is a common practical distinction for parameter-efficient fine-tuning because only a small set of adapter parameters is optimized.

The trainable methods use assistant-only loss masking. The system instruction and user prompt are context, but they do not contribute to the supervised loss; the optimization target is the assistant's POS-tag sequence.

## LoRA and QLoRA

For LoRA, the base Qwen model is loaded at BF16 when supported, otherwise FP16. The original model parameters are frozen and trainable low-rank matrices are attached to the attention and MLP projection layers:

```text
q_proj, k_proj, v_proj, o_proj
gate_proj, up_proj, down_proj
```

The adapter configuration is:

```text
rank (r)     = 16
alpha        = 32
dropout      = 0.05
bias         = none
```

QLoRA keeps the same adapter configuration but loads the base model with 4-bit NF4 quantization and double quantization. Computation still uses BF16 or FP16. This reduces the memory footprint of the frozen base model while preserving trainable LoRA adapters.

## Full fine-tuning vs. PEFT

Full fine-tuning updates the complete model. This gives the optimizer direct access to every parameter but has the highest memory cost.

LoRA freezes the base weights and learns low-rank updates. QLoRA takes the same idea one step further by storing the frozen base model in 4-bit precision.

The comparison is therefore useful for looking at the trade-off between task performance, trainable parameter count, training time, and GPU memory.

## Evaluation

The scripts report:

- **Accuracy**: token-level POS-tag accuracy.
- **Macro F1**: F1 averaged across the POS label set, giving each class equal weight.
- **Invalid output rate**: fraction of sentences where the generated answer cannot be accepted by the strict JSON/tag parser.
- **Training time** and **evaluation time**.
- **Peak CUDA memory reserved** during training and evaluation.

The invalid-output metric is important because this is a generative formulation of a structured prediction task. A model can know the correct tags but still fail the protocol by returning extra text, malformed JSON, or the wrong number of labels.

## Few-shot selection

Few-shot demonstrations are taken only from the training split. A fixed seed is used so the selected examples are reproducible, and very long training examples are excluded from the demonstration pool to keep the prompt manageable.

The same demonstrations are reused for every test sentence. No test labels are used to build the prompt.

## Reproducibility notes

The scripts set `SEED = 42` and use deterministic greedy decoding for generation. GPU model, CUDA version, PyTorch version, and library versions can still affect runtime and memory measurements, so comparisons should be run in the same environment when possible.

I intentionally do not hard-code benchmark numbers in this README. The scripts generate their own result CSVs, and those outputs should be treated as the source of truth for a particular run.

## Why the PyTorch LoRA version exists

`qwen_lora.py` is the concise implementation I would normally use for an experiment: TRL manages the supervised fine-tuning loop and PEFT manages the adapters.

`qwen_lora_pytorch.py` keeps PEFT for attaching the LoRA layers but implements the optimization loop manually. It is useful when I want direct control over, or need to inspect, the mechanics that a trainer normally hides.

That script explicitly handles:

```text
dataset -> DataLoader -> padded tensors
assistant-only label masking
forward pass
loss scaling
backpropagation
gradient accumulation
gradient clipping
AdamW update
cosine scheduler
validation loop
CUDA memory measurement
```

Keeping both versions makes the distinction between using a high-level training framework and understanding the underlying PyTorch workflow explicit.