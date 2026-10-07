import json
import time

import pandas as pd
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

SEED = 42
DATASET_NAME = "masakhane/masakhapos"
LANGUAGE_CODE = "yor"
LANGUAGE_NAME = "Yorùbá"
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./yoruba_pos_full_finetuning"
RESULTS_FILE = "qwen_full_finetune_results.csv"
TEST_MAX_SAMPLES = None

set_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected. Enable a GPU runtime before running this script.")

print("GPU:", torch.cuda.get_device_name(0))

bf16_supported = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
precision_name = "BF16" if bf16_supported else "FP16"

data = load_dataset(DATASET_NAME, LANGUAGE_CODE, trust_remote_code=True)

train_dataset = data["train"]
validation_dataset = data["validation"]
test_dataset = data["test"]

if TEST_MAX_SAMPLES is not None:
    test_dataset_for_eval = test_dataset.select(range(min(TEST_MAX_SAMPLES, len(test_dataset))))
else:
    test_dataset_for_eval = test_dataset

POS_LABELS = train_dataset.features["upos"].feature.names
allowed_tags = ", ".join(POS_LABELS)

SYSTEM_PROMPT = (
    f"You are a linguistic POS tagging system for {LANGUAGE_NAME}. "
    "Assign exactly one Universal POS tag to each input token. "
    f"Allowed POS tags are: {allowed_tags}. "
    "Preserve the token order. "
    "Return only a JSON list containing the POS tags."
)


def convert_upos_to_names(upos_values):
    return [POS_LABELS[int(tag)] for tag in upos_values]


def convert_to_chat(example):
    tokens = example["tokens"]
    pos_tags = convert_upos_to_names(example["upos"])

    user_message = (
        "Assign one UPOS tag to every token.\n\nTokens:\n"
        + json.dumps(tokens, ensure_ascii=False)
    )

    assistant_message = json.dumps(pos_tags, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


sft_train_dataset = train_dataset.map(
    convert_to_chat,
    remove_columns=train_dataset.column_names,
)

sft_validation_dataset = validation_dataset.map(
    convert_to_chat,
    remove_columns=validation_dataset.column_names,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=compute_dtype,
)

model.config.use_cache = False

total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)
trainable_percentage = trainable_parameters / total_parameters * 100

print("Total parameters:", f"{total_parameters:,}")
print("Trainable parameters:", f"{trainable_parameters:,}")
print("Trainable percentage:", f"{trainable_percentage:.4f}%")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    optim="adamw_torch",
    warmup_steps=10,
    lr_scheduler_type="cosine",
    bf16=bf16_supported,
    fp16=not bf16_supported,
    gradient_checkpointing=True,
    max_length=512,
    assistant_only_loss=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    report_to="none",
    seed=SEED,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_train_dataset,
    eval_dataset=sft_validation_dataset,
    processing_class=tokenizer,
)


def predict(tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Assign one UPOS tag to every token.\n\nTokens:\n"
            + json.dumps(tokens, ensure_ascii=False),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(trainer.model.device)
    max_new_tokens = min(512, max(64, len(tokens) * 8))

    with torch.inference_mode():
        outputs = trainer.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def parse_prediction(response, expected_length):
    try:
        prediction = json.loads(response)
    except json.JSONDecodeError:
        return None

    if not isinstance(prediction, list):
        return None

    if len(prediction) != expected_length:
        return None

    if any(tag not in POS_LABELS for tag in prediction):
        return None

    return prediction


def evaluate(dataset):
    all_gold = []
    all_predictions = []
    invalid_outputs = 0

    for index, example in enumerate(dataset):
        tokens = example["tokens"]
        gold_tags = convert_upos_to_names(example["upos"])

        response = predict(tokens)
        prediction = parse_prediction(response, len(tokens))

        if prediction is None:
            invalid_outputs += 1
            prediction = ["<INVALID>"] * len(gold_tags)

        all_gold.extend(gold_tags)
        all_predictions.extend(prediction)

        if (index + 1) % 100 == 0:
            print(f"Evaluated {index + 1}/{len(dataset)} sentences")

    accuracy = accuracy_score(all_gold, all_predictions)

    macro_f1 = f1_score(
        all_gold,
        all_predictions,
        labels=POS_LABELS,
        average="macro",
        zero_division=0,
    )

    invalid_output_rate = invalid_outputs / len(dataset)

    return accuracy, macro_f1, invalid_output_rate


trainer.model.config.use_cache = False

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

training_start = time.perf_counter()
trainer.train()
torch.cuda.synchronize()

training_time_seconds = time.perf_counter() - training_start
training_peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

validation_results = trainer.evaluate()
print("Validation results:", validation_results)

trainer.model.eval()
trainer.model.config.use_cache = True

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

evaluation_start = time.perf_counter()
accuracy, macro_f1, invalid_output_rate = evaluate(test_dataset_for_eval)
torch.cuda.synchronize()

evaluation_time_seconds = time.perf_counter() - evaluation_start
evaluation_peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nFull fine-tuning results")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Invalid output rate:", round(invalid_output_rate, 4))
print("Training time:", round(training_time_seconds / 60, 2), "minutes")
print("Evaluation time:", round(evaluation_time_seconds / 60, 2), "minutes")
print("Training peak VRAM:", round(training_peak_vram_gb, 2), "GB")
print("Evaluation peak VRAM:", round(evaluation_peak_vram_gb, 2), "GB")

results = pd.DataFrame(
    [
        {
            "Configuration": "Full fine-tuning",
            "Base precision": precision_name,
            "Trainable params": trainable_parameters,
            "Trainable %": round(trainable_percentage, 4),
            "Training peak VRAM (GB)": round(training_peak_vram_gb, 4),
            "Evaluation peak VRAM (GB)": round(evaluation_peak_vram_gb, 4),
            "Training time (min)": round(training_time_seconds / 60, 4),
            "Evaluation time (min)": round(evaluation_time_seconds / 60, 4),
            "Accuracy": round(accuracy, 4),
            "Macro F1": round(macro_f1, 4),
            "Invalid output rate": round(invalid_output_rate, 4),
        }
    ]
)

print("\n", results.to_string(index=False))
results.to_csv(RESULTS_FILE, index=False)
print("\nResults saved to:", RESULTS_FILE)