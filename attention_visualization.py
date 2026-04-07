"""
BERT Attention 可视化 —— 连接「文本分类」与「可解释性」

安装依赖：
    pip install bertviz matplotlib seaborn

用法：
    先跑完 bert_sentiment.py 保存模型，再运行本脚本
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

# ─── 0. 加载已训练的模型 ──────────────────────────────────────────────────────────
MODEL_DIR = "./bert_imdb_model"
tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)

# BertForSequenceClassification：用于推理预测标签
clf_model = BertForSequenceClassification.from_pretrained(MODEL_DIR, output_attentions=True)
clf_model.eval()

# ─── 1. 选几条典型样本 ────────────────────────────────────────────────────────────
samples = [
    "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
    "Terrible film. Boring, predictable, and a complete waste of time. I fell asleep halfway through.",
    "The movie had some great moments but overall felt a bit too long and the ending disappointed me.",
]

# ─── 2. 单样本推理 + 拿到 attention weights ───────────────────────────────────────
def predict_and_get_attention(text):
    """
    返回:
        label     : 预测标签 (0=negative, 1=positive)
        confidence: softmax 概率
        tokens    : 分词结果（含 [CLS] / [SEP]）
        attentions: tuple of (num_layers,) 每层 shape=(1, num_heads, seq, seq)
    """
    enc = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
    with torch.no_grad():
        output = clf_model(**enc)

    probs      = torch.softmax(output.logits, dim=-1)[0]
    label      = torch.argmax(probs).item()
    confidence = probs[label].item()

    # 只取非 padding 的真实 token 数量
    real_len   = enc["attention_mask"][0].sum().item()
    tokens     = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])[:real_len]

    # attentions: 每一层的 shape (1, heads, seq, seq) → 取 [0] 去掉 batch 维
    attentions = [a[0].detach().numpy() for a in output.attentions]

    return label, confidence, tokens, attentions, real_len

# ─── 3. 可视化一：热力图（某一层、某个 Head 的 Attention 矩阵）─────────────────────
def plot_attention_heatmap(tokens, attention_matrix, layer, head, title_extra=""):
    """
    attention_matrix: shape (num_heads, seq, seq)
    """
    real_len = len(tokens)
    attn = attention_matrix[head][:real_len, :real_len]   # 裁到真实长度

    fig, ax = plt.subplots(figsize=(max(6, real_len * 0.45), max(5, real_len * 0.4)))
    sns.heatmap(
        attn,
        xticklabels = tokens,
        yticklabels = tokens,
        cmap        = "Blues",
        vmin        = 0, vmax = attn.max(),
        ax          = ax,
        linewidths  = 0.3,
        linecolor   = "white"
    )
    ax.set_title(f"Layer {layer+1}  Head {head+1} · Attention Weights\n{title_extra}", fontsize=11)
    ax.set_xlabel("Key tokens (被关注)")
    ax.set_ylabel("Query tokens (正在关注)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fname = f"attention_L{layer+1}_H{head+1}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"已保存：{fname}")

# ─── 4. 可视化二：CLS token 跨层平均 Attention（模型关注的 token 排名）──────────────
def plot_cls_attention_ranking(tokens, attentions, real_len, title=""):
    """
    [CLS] token 的 attention 代表模型做分类时「看了哪些词」。
    对所有层、所有 head 取平均，看哪些 token 最受关注。
    """
    # attentions: list of (num_heads, seq, seq), len=num_layers
    # [CLS] 在位置 0，取第 0 行
    cls_attns = []
    for layer_attn in attentions:
        # layer_attn: (heads, seq, seq) → 取所有 head 对 CLS 行的均值
        cls_attn = layer_attn[:, 0, :real_len].mean(axis=0)  # (real_len,)
        cls_attns.append(cls_attn)

    avg_cls_attn = np.stack(cls_attns).mean(axis=0)   # 跨层平均 (real_len,)
    avg_cls_attn = avg_cls_attn / avg_cls_attn.sum()  # 归一化

    # 跳过 [CLS] 和 [SEP] 本身，只看内容 token
    content_tokens = tokens[1:-1]
    content_attn   = avg_cls_attn[1:real_len-1]

    sorted_idx  = np.argsort(content_attn)[::-1]
    top_tokens  = [content_tokens[i] for i in sorted_idx[:10]]
    top_weights = [content_attn[i] for i in sorted_idx[:10]]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(top_tokens[::-1], top_weights[::-1], color="steelblue", alpha=0.8)
    ax.set_xlabel("平均 Attention 权重（CLS → token）")
    ax.set_title(f"模型最关注的 Top-10 词\n{title}", fontsize=11)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    fname = "cls_attention_ranking.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"已保存：{fname}")

# ─── 5. 可视化三：跨层 Attention 演变（同一 Head，看不同层如何聚焦）──────────────────
def plot_attention_evolution(tokens, attentions, real_len, head=0):
    """
    选择某个 Head，画出它在 1/4/8/12 层的 [CLS] attention 分布变化。
    """
    num_layers = len(attentions)
    layers_to_show = [0, num_layers//4, num_layers//2, num_layers-1]

    fig, axes = plt.subplots(1, len(layers_to_show), figsize=(14, 3), sharey=True)
    content_tokens = tokens[1:real_len-1]

    for ax, layer_idx in zip(axes, layers_to_show):
        attn = attentions[layer_idx][head, 0, 1:real_len-1]   # CLS → content tokens
        attn = attn / attn.sum()
        ax.bar(range(len(content_tokens)), attn, color="coral", alpha=0.8)
        ax.set_xticks(range(len(content_tokens)))
        ax.set_xticklabels(content_tokens, rotation=60, ha="right", fontsize=7)
        ax.set_title(f"Layer {layer_idx+1}", fontsize=10)
        ax.set_ylim(0, 0.5)

    axes[0].set_ylabel("Attention 权重")
    fig.suptitle(f"Head {head+1} 跨层 Attention 演变（[CLS] → content tokens）", fontsize=11)
    plt.tight_layout()
    fname = f"attention_evolution_H{head+1}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"已保存：{fname}")

# ─── 6. 主流程：跑三个样本 ───────────────────────────────────────────────────────────
label_map = {0: "negative ❌", 1: "positive ✅"}

for i, text in enumerate(samples):
    print(f"\n{'='*60}")
    print(f"样本 {i+1}: {text[:80]}...")

    label, conf, tokens, attentions, real_len = predict_and_get_attention(text)
    print(f"预测：{label_map[label]}  置信度：{conf:.3f}")
    print(f"Tokens: {tokens}")

    # 热力图：第 12 层（最后一层），Head 1
    plot_attention_heatmap(
        tokens, attentions[-1],
        layer=11, head=0,
        title_extra=f"预测={label_map[label]} ({conf:.2f})"
    )

    # CLS attention 排名
    plot_cls_attention_ranking(tokens, attentions, real_len, title=text[:50]+"...")

    # 跨层演变
    plot_attention_evolution(tokens, attentions, real_len, head=0)

# ─── 7. 彩蛋：用 bertviz 做交互式可视化（可选）────────────────────────────────────────
print("""
【可选】BertViz 交互式可视化 (Jupyter Notebook 中运行)：

    from bertviz import head_view
    from transformers import BertModel, BertTokenizer

    model = BertModel.from_pretrained('./bert_imdb_model', output_attentions=True)
    tokenizer = BertTokenizer.from_pretrained('./bert_imdb_model')

    text = "This movie was fantastic!"
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)

    head_view(outputs.attentions, tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

运行后会在 Notebook 中弹出可交互的 head-level attention 可视化图。
""")
