import numpy as np
import pandas as pd
import torch
import time
import os
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

LABEL_CLASSES = ["NORMAL", "HIGH_TEMP", "HIGH_HUMIDITY", "GAS_ANOMALY", "COMBINED_STRESS"]
MODEL_NAME    = "sshleifer/tiny-distilbert-base-cased"
BATCH_SIZE    = 4
EPOCHS        = 1
MAX_LEN       = 32   # short inputs → fast inference
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SensorDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


def load_data(csv_path="agriculture_dataset.csv"):
    df  = pd.read_csv(csv_path)
    le  = LabelEncoder()
    le.fit(LABEL_CLASSES)
    y   = le.transform(df["label"].values)
    return df["text_input"].tolist(), y, le


def train_distilbert(csv_path="agriculture_dataset.csv", save_dir="saved_models/distilbert"):
    print(f"Device: {DEVICE}")
    texts, y, le = load_data(csv_path)

    X_train_t, X_test_t, y_train, y_test = train_test_split(
        texts, y, test_size=0.2, random_state=42, stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    train_ds  = SensorDataset(X_train_t, y_train, tokenizer)
    test_ds   = SensorDataset(X_test_t,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_CLASSES),
    ignore_mismatched_sizes=True
).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    scheduler = get_scheduler(
        "linear", optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_loader) * EPOCHS
    )

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss           = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    latencies = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start          = time.perf_counter()
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            latencies.append((time.perf_counter() - start) * 1000 / len(batch["labels"]))
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch["labels"].numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="weighted")
    avg_lat = np.mean(latencies)

    print(f"\nTinyBERT SLM Results:")
    print(f"  Accuracy:      {acc:.4f}")
    print(f"  F1 (weighted): {f1:.4f}")
    print(f"  Avg Latency:   {avg_lat:.2f} ms/sample")
    print(classification_report(
    all_labels,
    all_preds,
    labels=list(range(len(LABEL_CLASSES))),
    target_names=LABEL_CLASSES,
    zero_division=0
))

    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model saved to {save_dir}")

    return {
        "model":          "Tiny DistilBERT",
        "accuracy":       round(acc, 4),
        "f1_weighted":    round(f1, 4),
        "avg_latency_ms": round(avg_lat, 3),
        "p95_latency_ms": round(np.percentile(latencies, 95), 3),
        "model_size_kb":  sum(
            os.path.getsize(os.path.join(save_dir, f)) / 1024
            for f in os.listdir(save_dir) if os.path.isfile(os.path.join(save_dir, f))
        )
    }


def export_to_onnx(model_dir="saved_models/distilbert", output_path="saved_models/distilbert.onnx"):
    """
    Export to ONNX for faster inference on Raspberry Pi.
    On Pi: use onnxruntime instead of torch for 2-3x speedup.
    """
    from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model     = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    dummy_input = tokenizer(
        "temperature 25.0 humidity 60.0 gas 400",
        return_tensors="pt", padding="max_length",
        max_length=MAX_LEN, truncation=True
    )

    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
        opset_version=12
    )
    print(f"ONNX model exported to {output_path}")
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"ONNX model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    results = train_distilbert(csv_path="agriculture_dataset.csv")
    print("\nFinal metrics:", results)
    # Uncomment to export for Pi deployment:
    # export_to_onnx()
