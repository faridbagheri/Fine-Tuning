import json
import math
import time

import pandas as pd
import torch

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, set_seed

SEED = 42
DATASET_NAME = "masakhane/masakhapos"
LANGUAGE_CODE = "yor"
LANGUAGE_NAME = "Yorùbá"
MODEL_NAME = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "./yoruba_pos_lora_pytorch"
RESULTS_FILE = "qwen_lora_pytorch_results.csv"
TEST_MAX_SAMPLES = None

EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
MAX_LENGTH = 512

set_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("GPU not detected. Enable a GPU runtime before running this script.")

device = torch.device("cuda")
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

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": json.dumps(pos_tags, ensure_ascii=False)},
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


def encode_example(example):
    messages = example["messages"]
    prompt_messages = messages[:-1]

    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )

    input_ids = input_ids[:MAX_LENGTH]
    prompt_length = min(len(prompt_ids), len(input_ids))

    labels = input_ids.copy()

    for index in range(prompt_length):
        labels[index] = -100

    return {
        "input_ids": input_ids,
        "labels": labels,
    }


def collate_batch(examples):
    encoded = [encode_example(example) for example in examples]
    batch_size = len(encoded)
    max_length = max(len(example["input_ids"]) for example in encoded)

    input_ids = torch.full(
        (batch_size, max_length),
        tokenizer.pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
    )

    labels = torch.full(
        (batch_size, max_length),
        -100,
        dtype=torch.long,
    )

    for row, example in enumerate(encoded):
        length = len(example["input_ids"])
        input_ids[row, :length] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(example["labels"], dtype=torch.long)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

train_loader = DataLoader(
    sft_train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_batch,
    generator=loader_generator,
    pin_memory=True,
)

validation_loader = DataLoader(
    sft_validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch,
    pin_memory=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=compute_dtype,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.to(device)

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

optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=LEARNING_RATE,
)

updates_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
total_update_steps = updates_per_epoch * EPOCHS

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_update_steps,
)

scaler = torch.cuda.amp.GradScaler(enabled=not bf16_supported)


def move_batch_to_device(batch):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def train_one_epoch(epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0

    for step, batch in enumerate(train_loader):
        batch = move_batch_to_device(batch)

        with torch.autocast(
            device_type="cuda",
            dtype=compute_dtype,
        ):
            outputs = model(**batch)
            loss = outputs.loss
            scaled_loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(scaled_loss).backward()
        running_loss += loss.detach().float().item()

        should_update = (
            (step + 1) % GRAD_ACCUM_STEPS == 0
            or (step + 1) == len(train_loader)
        )

        if should_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        if (step + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Batch {step + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    return running_loss / len(train_loader)


def validation_loss():
    model.eval()
    running_loss = 0.0

    with torch.inference_mode():
        for batch in validation_loader:
            batch = move_batch_to_device(batch)

            with torch.autocast(
                device_type="cuda",
                dtype=compute_dtype,
            ):
                outputs = model(**batch)

            running_loss += outputs.loss.detach().float().item()

    return running_loss / len(validation_loader)


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

    inputs = tokenizer(text, return_tensors="pt").to(device)
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
    model.eval()
    model.config.use_cache = True

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

training_start = time.perf_counter()

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(epoch)
    val_loss = validation_loss()
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train loss = {train_loss:.4f} | "
        f"validation loss = {val_loss:.4f}"
    )

torch.cuda.synchronize()

training_time_seconds = time.perf_counter() - training_start
training_peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

model.eval()
model.config.use_cache = True

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

evaluation_start = time.perf_counter()
accuracy, macro_f1, invalid_output_rate = evaluate(test_dataset_for_eval)
torch.cuda.synchronize()

evaluation_time_seconds = time.perf_counter() - evaluation_start
evaluation_peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nManual PyTorch LoRA results")
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
            "Configuration": "LoRA",
            "Implementation": "Manual PyTorch training loop + PEFT",
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
