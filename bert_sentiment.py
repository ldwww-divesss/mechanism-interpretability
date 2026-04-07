"""
BERT Fine-tune for IMDB Sentiment Classification
Kaggle: https://www.kaggle.com/c/word2vec-nlp-tutorial/

环境安装：
    pip install transformers datasets torch scikit-learn

数据准备：
    从 Kaggle 下载 labeledTrainData.tsv 和 testData.tsv
    放到当前目录即可
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

# ─── 0. 配置 ───────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "/home/user/tiny_bert"   # 本地小模型（CPU 演示用）；替换为 "bert-base-uncased" 可用完整模型
MAX_LEN    = 64    # 演示缩短（完整训练建议 256）
BATCH_SIZE = 8
EPOCHS     = 1     # 演示 1 轮（完整训练建议 3）
LR         = 2e-5

print(f"Using device: {DEVICE}")

# ─── 1. 数据加载与清洗 ──────────────────────────────────────────────────────────
def clean_text(text):
    """去掉 HTML 标签和多余空格"""
    text = re.sub(r"<.*?>", " ", text)       # 移除 <br /> 等 HTML 标签
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv("/home/user/mechanism-interpretability/labeledTrainData.tsv", sep="\t", quoting=3)
df["review"] = df["review"].apply(clean_text)

# 划分训练集 / 验证集（80/20）
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment"])
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

# ─── 2. Dataset ─────────────────────────────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt"
        )
        return {
            "input_ids"      : enc["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask" : enc["attention_mask"].squeeze(0),
            "label"          : torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = IMDBDataset(train_df["review"], train_df["sentiment"], tokenizer, MAX_LEN)
val_dataset   = IMDBDataset(val_df["review"],   val_df["sentiment"],   tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)

# ─── 3. 模型 ────────────────────────────────────────────────────────────────────
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# ─── 4. 优化器 + 学习率调度 ──────────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps   = total_steps // 10,   # 前 10% 步做 warmup
    num_training_steps = total_steps
)

# ─── 5. 训练循环 ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪，防止梯度爆炸
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            preds_all.extend(preds)
            labels_all.extend(labels.numpy())

    return accuracy_score(labels_all, preds_all), classification_report(labels_all, preds_all, target_names=["negative","positive"])

for epoch in range(EPOCHS):
    avg_loss = train_epoch(model, train_loader, optimizer, scheduler)
    acc, report = evaluate(model, val_loader)
    print(f"\nEpoch {epoch+1}/{EPOCHS}  |  Loss: {avg_loss:.4f}  |  Val Acc: {acc:.4f}")
    print(report)

# ─── 6. 保存模型 ─────────────────────────────────────────────────────────────────
model.save_pretrained("/home/user/mechanism-interpretability/bert_imdb_model")
tokenizer.save_pretrained("/home/user/mechanism-interpretability/bert_imdb_model")
print("\n模型已保存到 ./bert_imdb_model")
