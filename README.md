
## 文件说明
- `bert_sentiment.py`      — BERT fine-tune 训练脚本
- `attention_visualization.py` — Attention 可视化脚本

---

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

## Attention 可视化

### 1. 安装额外依赖
```bash
pip install matplotlib seaborn bertviz
```

### 2. 运行可视化
```bash
python attention_visualization.py
```

| 图 | 文件名 | 说明 |
|----|--------|------|
| Attention 热力图 | `attention_L12_H1.png` | 第12层第1头，token间注意力矩阵 |
| CLS 关注排名 | `cls_attention_ranking.png` | 模型分类时最关注哪些词 |
| 跨层演变 | `attention_evolution_H1.png` | 同一 Head 在各层的关注点变化 |

---


