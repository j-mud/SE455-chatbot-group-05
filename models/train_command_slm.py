import os
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report


# command intent Labels

COMMAND_LABELS = [
    "EXPLAIN_PUMP",
    "CHECK_SOIL",
    "CHECK_TEMP",
    "CHECK_HUMIDITY",
    "CHECK_GAS",
    "GENERAL_STATUS",
]

MODEL_NAME = "sshleifer/tiny-distilbert-base-cased"
SAVE_DIR = "saved_models/command_slm"

EPOCHS = 10
BATCH_SIZE = 4
MAX_LEN = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Local Command Dataset ;/ 36

examples = [
    ("why is the pump on", "EXPLAIN_PUMP"),
    ("why is pump off", "EXPLAIN_PUMP"),
    ("explain the pump status", "EXPLAIN_PUMP"),
    ("is irrigation active", "EXPLAIN_PUMP"),
    ("why is the system watering", "EXPLAIN_PUMP"),
    ("should the pump run", "EXPLAIN_PUMP"),

    ("check soil", "CHECK_SOIL"),
    ("what is the soil moisture", "CHECK_SOIL"),
    ("is the soil dry", "CHECK_SOIL"),
    ("soil status", "CHECK_SOIL"),
    ("how wet is the soil", "CHECK_SOIL"),
    ("do we need water", "CHECK_SOIL"),

    ("check temperature", "CHECK_TEMP"),
    ("what is the temperature", "CHECK_TEMP"),
    ("is it hot", "CHECK_TEMP"),
    ("temperature status", "CHECK_TEMP"),
    ("heat level", "CHECK_TEMP"),
    ("is the greenhouse hot", "CHECK_TEMP"),

    ("check humidity", "CHECK_HUMIDITY"),
    ("what is the humidity", "CHECK_HUMIDITY"),
    ("is it humid", "CHECK_HUMIDITY"),
    ("humidity level", "CHECK_HUMIDITY"),
    ("moisture in air", "CHECK_HUMIDITY"),
    ("air humidity status", "CHECK_HUMIDITY"),

    ("check gas", "CHECK_GAS"),
    ("what is the gas level", "CHECK_GAS"),
    ("air quality status", "CHECK_GAS"),
    ("is the air safe", "CHECK_GAS"),
    ("gas sensor reading", "CHECK_GAS"),
    ("check air quality", "CHECK_GAS"),

    ("overall status", "GENERAL_STATUS"),
    ("system status", "GENERAL_STATUS"),
    ("how is the greenhouse", "GENERAL_STATUS"),
    ("give me summary", "GENERAL_STATUS"),
    ("what is happening", "GENERAL_STATUS"),
    ("analyze current condition", "GENERAL_STATUS"),
]


class CommandDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "labels": self.labels[index],
        }


def train_command_slm():
    print(f"Device: {DEVICE}")

    label_to_id = {label: i for i, label in enumerate(COMMAND_LABELS)}
    id_to_label = {i: label for i, label in enumerate(COMMAND_LABELS)}

    texts = [x[0] for x in examples]
    labels = [label_to_id[x[1]] for x in examples]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_dataset = CommandDataset(X_train, y_train, tokenizer)
    test_dataset = CommandDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(COMMAND_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels_tensor = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_tensor,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    all_preds = []
    all_labels = []
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            start = time.perf_counter()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            latency = (time.perf_counter() - start) * 1000 / len(batch["labels"])
            latencies.append(latency)

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\nCommand SLM Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Weighted: {f1:.4f}")
    print(f"Average Latency: {np.mean(latencies):.4f} ms/sample")
    print(f"P95 Latency: {np.percentile(latencies, 95):.4f} ms/sample")

    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(len(COMMAND_LABELS))),
            target_names=COMMAND_LABELS,
            zero_division=0,
        )
    )

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    model_size_kb = sum(
        os.path.getsize(os.path.join(SAVE_DIR, file)) / 1024
        for file in os.listdir(SAVE_DIR)
        if os.path.isfile(os.path.join(SAVE_DIR, file))
    )

    print(f"Command SLM saved to {SAVE_DIR}")

    return {
        "model": "Tiny DistilBERT Command SLM",
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
        "model_size_kb": round(model_size_kb, 2),
    }


if __name__ == "__main__":
    results = train_command_slm()
    print("\nFinal command SLM metrics:", results)