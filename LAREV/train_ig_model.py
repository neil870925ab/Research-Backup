import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import os


class ESNLIRationaleDataset(Dataset):
    """
    Reads baseline_rationales_train_output.jsonl

    Expected format per line:
    {
        "question_text": "...",
        "answer_text": "entailment / contradiction / neutral",
        "question_statement_text": "<paraphrased baseline rationale>"
    }
    """
    LABEL_MAP = {
        "entailment": 0,
        "contradiction": 1,
        "neutral": 2,
    }

    def __init__(self, jsonl_path, tokenizer, max_length=256):
        self.samples = []
        with open(jsonl_path, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["question_statement_text"]
                label = self.LABEL_MAP[obj["answer_text"]]
                self.samples.append((text, label))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        return item


def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = out.loss
        logits = out.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / (len(dataloader.dataset))

    print(f"[Train] loss = {avg_loss:.4f}, acc = {accuracy:.4f}")
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="ESNLI")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        f"cuda:{args.device}"
        if torch.cuda.is_available() and args.device != "cpu"
        else "cpu"
    )

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
        print("Directory '% s' created" % args.out_dir)

    current_path = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_path, '../', '../', 'generated_baseline_rationales', args.task, 'baseline_rationales_train_output.jsonl')
    train_file = os.path.normpath(train_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    if args.task.lower() == "esnli":
        dataset = ESNLIRationaleDataset(
            jsonl_path=train_file,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)

    # -----------------------------
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        print(f"========== Epoch {epoch+1}/{args.epochs} ==========")
        train(model, dataloader, optimizer, device)


    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print(f"IG probe model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
