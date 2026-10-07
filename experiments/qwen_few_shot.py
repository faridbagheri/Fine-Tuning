import json
import time

import pandas as pd
import torch

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

SEED = 42
DATASET_NAME = "masakhane/masakhapos"
LANGUAGE_CODE = "yor"
LANGUAGE_NAME = "Yorùbá"
MODEL_NAME = "Qwen/Qwen3-0.6B"
K_SHOTS = 5
MAX_DEMO_TOKENS = 30
TEST_MAX_SAMPLES = None
RESULTS_FILE = f"qwen_yoruba_{K_SHOTS}_shot_results.csv"

set_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected. Enable a GPU runtime before running this script.")

print("GPU:", torch.cuda.get_device_name(0))

bf16_supported = torch.cuda.is_bf16_supported()
compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
precision_name = "BF16" if bf16_supported else "FP16"

data = load_dataset(DATASET_NAME, LANGUAGE_CODE, trust_remote_code=True)

train_dataset = data["train"]
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


def make_user_message(tokens):
    return (
        "Assign one UPOS tag to every token.\n\nTokens:\n"
        + json.dumps(tokens, ensure_ascii=False)
    )


candidate_indices = [
    index
    for index, example in enumerate(train_dataset)
    if len(example["tokens"]) <= MAX_DEMO_TOKENS
]

if len(candidate_indices) < K_SHOTS:
    raise ValueError(
        f"Only {len(candidate_indices)} suitable training examples were found, "
        f"but K_SHOTS={K_SHOTS}."
    )

generator = torch.Generator()
generator.manual_seed(SEED)

permutation = torch.randperm(
    len(candidate_indices),
    generator=generator,
).tolist()

selected_demo_indices = [
    candidate_indices[index]
    for index in permutation[:K_SHOTS]
]

few_shot_examples = [
    train_dataset[index]
    for index in selected_demo_indices
]

print("Selected few-shot training indices:", selected_demo_indices)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=compute_dtype,
).to("cuda")

model.eval()
model.config.use_cache = True


def build_messages(tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    for example in few_shot_examples:
        demo_tokens = example["tokens"]
        demo_tags = convert_upos_to_names(example["upos"])

        messages.append(
            {
                "role": "user",
                "content": make_user_message(demo_tokens),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(demo_tags, ensure_ascii=False),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": make_user_message(tokens),
        }
    )

    return messages


def predict(tokens):
    messages = build_messages(tokens)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    max_new_tokens = min(512, max(64, len(tokens) * 8))

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


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


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

start_time = time.perf_counter()
accuracy, macro_f1, invalid_output_rate = evaluate(test_dataset_for_eval)
torch.cuda.synchronize()

evaluation_time_seconds = time.perf_counter() - start_time
peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

print(f"\n{K_SHOTS}-shot results")
print("Accuracy:", round(accuracy, 4))
print("Macro F1:", round(macro_f1, 4))
print("Invalid output rate:", round(invalid_output_rate, 4))
print("Evaluation time:", round(evaluation_time_seconds / 60, 2), "minutes")
print("Peak VRAM:", round(peak_vram_gb, 2), "GB")

results = pd.DataFrame(
    [
        {
            "Configuration": f"Few-shot ({K_SHOTS}-shot)",
            "Base precision": precision_name,
            "Trainable params": 0,
            "Trainable %": 0.0,
            "Peak VRAM (GB)": round(peak_vram_gb, 4),
            "Training time (min)": None,
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