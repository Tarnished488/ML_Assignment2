# ML_Assignment2
Machine learning group assignment 2 — semi-supervised learning on limited labeled data.

## Team Division

| Name | Role | Responsibilities |
|------|------|-----------------|
| Qingyuan You | Data Preprocessing & Simple Models | Data loading, train/val split, normalization, baseline models |
| Yuxin Tang | Complex Model Development | Stronger supervised models, stable and well-generalizing |
| Yalu Wang | Semi-Supervised Learning | Leverage unlabeled data to improve classification with limited labels |
| Wanting Li | Visualization & Experiment Pipeline | Visual analytics, decision boundaries, experiment workflow, report analysis |

---

## Project Structure

```
ML_Assignment2/
├── train_mlp.py              # 完整训练流水线 (GitHub最新版)
├── train_final.py            # 多seed投票融合，高分作品保留
├── train_mlp128.py           # MLP 512→128→128→10 (正在跑)
├── src/
│   ├── models/
│   │   ├── mlp.py            # 残差MLP (512→256→128→64→10)
│   │   └── cnn.py            # CNN模型
│   ├── ssl/
│   │   ├── label_propagation.py   # k-NN图标签传播
│   │   ├── clustering.py          # 聚类标签广播
│   │   ├── self_training.py      # 迭代自训练
│   │   └── consistency.py        # VAT一致性正则化
│   ├── preprocessing/
│   │   └── data_loader.py    # 数据加载
│   ├── visualization/
│   │   └── visualizer.py     # 可视化图表
│   └── utils.py              # 共享工具函数
└── archive/
    └── semi_supervised.py.bak  # 归档旧代码
```

---

## 模型架构

### MLP (src/models/mlp.py)
- 残差连接 ResidualBlock
- 支持 BatchNorm / LayerNorm
- 支持 GELU / ReLU / SiLU 激活函数
- 三个工厂函数：`build_mlp`, `build_mlp_deep`, `build_mlp_wide`

### CNN (src/models/cnn.py)
- `build_cnn_32x16`: 32×16布局
- `build_cnn_8x64`: 8×64布局

---

## 半监督学习方法

### 1. Label Propagation (label_propagation.py)
k-NN图标签传播，基于特征空间相似性传播标签。对100条标注数据效果显著。
```
python train_mlp.py --name mlp_lp --use-ssl --ssl-method distill
```

### 2. Knowledge Distillation (train_mlp.py)
LP软概率作为teacher，MLP做student，避免硬伪标签丢失类别模糊信息。
```
python train_mlp.py --name mlp_distill --use-ssl --ssl-method distill
```

### 3. Self-Training (self_training.py)
多轮迭代自训练，动态阈值+每轮伪标签上限。
```
python train_mlp.py --name mlp_st --use-ssl --ssl-method self_training --self-train-rounds 5
```

### 4. Clustering (clustering.py)
KMeans++聚类 + 标签广播，支持约束聚类(ConstrainedClusteringSSL)。
```
python train_mlp.py --name mlp_cluster --use-ssl --ssl-method clustering
```

### 5. VAT (consistency.py)
Virtual Adversarial Training 一致性正则化，配合distill使用效果更佳。
```
python train_mlp.py --name mlp_vat --use-ssl --ssl-method distill --use-vat
```

---

## 快速开始

### 完整流水线训练
```bash
# 知识蒸馏 (推荐 ~0.5+ 准确率)
python train_mlp.py --name mlp_distill --use-ssl --ssl-method distill

# 聚类标签广播
python train_mlp.py --name mlp_cluster --use-ssl --ssl-method clustering

# 迭代自训练
python train_mlp.py --name mlp_st --use-ssl --ssl-method self_training --self-train-rounds 5

# 纯监督 (只用100条标注数据)
python train_mlp.py --name mlp_supervised
```

### 主要参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--epochs` | 训练轮数 | 200 |
| `--batch-size` | 批大小 | 32 |
| `--lr` | 学习率 | 1e-3 |
| `--hidden-dims` | MLP隐藏层维度 | 256,128,64 |
| `--dropout` | Dropout率 | 0.3 |
| `--ssl-method` | SSL方法 | distill |
| `--val-size` | 验证集比例 | 0.2 |

### Label Propagation 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lp-k` | k-NN邻居数 | 10 |
| `--lp-alpha` | 置信度因子 | 0.99 |
| `--lp-top-k` | 每类伪标签上限 | 300 |

### Self-Training 参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--self-train-rounds` | 迭代轮数 | 2 |
| `--self-train-threshold` | 置信度阈值 | 0.85 |
| `--st-dynamic-threshold` | 动态阈值开关 | True |
| `--st-initial-threshold` | 起始阈值 | 0.95 |
| `--st-top-k-per-round` | 每轮伪标签上限 | 500 |

---

## 输出

训练结果保存在 `outputs/` 目录：
- `submission.csv` - Kaggle提交文件
- `model.pt` - 训练好的模型权重
- `test_probs.npy` - 测试集预测概率

可视化输出 (plots/):
- PCA / t-SNE 可视化
- 混淆矩阵
- 决策边界
- 损失曲面
- 训练曲线

---

## 数据

- 标注数据：100条，512维特征，10类
- 无标注数据：~7000条
- 测试集：8000条

数据路径可通过环境变量 `ASSIGNMENT2_DATA_DIR` 或 `--data-dir` 参数指定。