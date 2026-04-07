# 1天搞定 BERT 文本分类 + 可解释性

## 文件说明
- `bert_sentiment.py`      — BERT fine-tune 训练脚本（上午）
- `attention_visualization.py` — Attention 可视化脚本（下午）

---

## 上午：跑通训练（约 3 小时）

### 1. 安装依赖
```bash
pip install transformers datasets torch scikit-learn pandas
```

### 2. 下载数据
从 Kaggle 下载并解压：
https://www.kaggle.com/c/word2vec-nlp-tutorial/data

把 `labeledTrainData.tsv` 放到脚本同目录。

### 3. 运行训练
```bash
python bert_sentiment.py
```

**预期结果：**
- Epoch 1: Val Acc ≈ 0.90+
- Epoch 3: Val Acc ≈ 0.93~0.94
- 训练时间：有 GPU 约 20 分钟，纯 CPU 约 2~3 小时（建议用 Colab）

### 4. 用 Colab 省时间（推荐）
打开 Google Colab → 选 GPU 运行时 → 上传脚本和数据 → 运行

---

## 下午：Attention 可视化（约 2 小时）

### 1. 安装额外依赖
```bash
pip install matplotlib seaborn bertviz
```

### 2. 运行可视化
```bash
python attention_visualization.py
```

**会生成三类图：**

| 图 | 文件名 | 说明 |
|----|--------|------|
| Attention 热力图 | `attention_L12_H1.png` | 第12层第1头，token间注意力矩阵 |
| CLS 关注排名 | `cls_attention_ranking.png` | 模型分类时最关注哪些词 |
| 跨层演变 | `attention_evolution_H1.png` | 同一 Head 在各层的关注点变化 |

---

## 面试时怎么说

**被问到「做过什么 NLP 项目」时：**

> 用 BERT fine-tune 做了 IMDB 情感分类，验证集准确率 93%+。
> 之后在模型上做了 Attention 可视化分析 —— 发现最后一层的 [CLS] token
> 对「fantastic」「terrible」等情感词的 attention 权重显著高于其他词，
> 并且观察到浅层 attention 分散（句法层面），深层 attention 集中（语义层面），
> 这和 Mechanistic Interpretability 中关于层级功能分工的发现一致。

这句话直接把分类任务和可解释性研究串联起来。

---

## 如果时间不够

只跑训练，不跑可视化，至少要能回答：

1. BERT 的 [CLS] token 为什么能代表整句语义？
   → 因为它参与了所有层的 self-attention，聚合了全句上下文信息。

2. fine-tune 时为什么学习率设 2e-5 而不是 1e-3？
   → BERT 预训练权重已经很好，太大的 lr 会破坏（灾难性遗忘），小 lr 做微调。

3. 为什么做 warmup？
   → 训练初期参数变化大，warmup 让 lr 从 0 缓慢增加，避免早期不稳定更新破坏预训练特征。
