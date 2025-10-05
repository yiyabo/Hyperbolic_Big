# PRING基准测试数据集使用指南

**版本**: 1.0  
**日期**: 2025年10月4日  
**来源**: [PRING: Rethinking PPI Prediction from Pairs to Graphs](https://arxiv.org/abs/2507.05101) (NeurIPS 2025)

---

## 📋 概述

PRING是一个标准化的蛋白质相互作用预测基准测试，从**成对预测**转向**网络级评估**。本项目采用PRING作为主要训练和评估数据集。

## 🎯 为什么选择PRING？

### 1. **标准化评估**
- NeurIPS 2025官方基准测试
- 可与其他方法公平对比
- 学术界认可的评估标准

### 2. **高质量数据**
- ✅ 序列长度过滤（50-1000aa）
- ✅ 序列相似度过滤（MMseqs2，阈值0.4）
- ✅ 功能相似性过滤
- ✅ 拓扑驱动的负样本采样
- ✅ 严格的train/val/test切分

### 3. **网络级视角**
- 不仅评估成对预测准确率
- 评估重建网络的拓扑质量
- 评估生物学功能可解释性

### 4. **工程友好**
- 数据预处理完成
- 提供标准化评估代码
- 支持多种实验设置

## 📂 数据集结构

```
data/PRING/data_process/pring_dataset/
├── human/                          # 人类（训练集）
│   ├── BFS/                       # 广度优先采样策略
│   │   ├── human_train_ppi.txt   # 训练PPI对
│   │   ├── human_val_ppi.txt     # 验证PPI对
│   │   ├── human_test_ppi.txt    # 测试PPI对（二分类）
│   │   ├── all_test_ppi.txt      # 测试PPI对（图重建）
│   │   ├── human_train_graph.pkl # 训练图
│   │   ├── human_test_graph.pkl  # 测试图（真实标签）
│   │   └── test_sampled_nodes.pkl # BFS采样的测试子图
│   ├── DFS/                       # 深度优先采样策略
│   └── RANDOM_WALK/               # 随机游走采样策略
│   ├── human_graph.pkl            # 完整PPI图
│   ├── human_ppi.txt              # 所有PPI对
│   ├── human_protein_id.csv       # 蛋白质ID映射
│   ├── human_simple.fasta         # 蛋白质序列（UniProt ID）
│   └── human.fasta                # 蛋白质序列（完整meta）
├── arath/                          # 拟南芥（跨物种测试）
├── yeast/                          # 酵母（跨物种测试）
└── ecoli/                          # 大肠杆菌（跨物种测试）
```

## 📊 数据统计

### Human（训练集）
- **蛋白质**: ~19,000个
- **训练PPI**: ~100,000对
- **验证PPI**: ~10,000对
- **测试PPI**: ~10,000对
- **采样策略**: BFS, DFS, Random Walk（3种）

### 跨物种测试集
| 物种 | 蛋白质数 | 测试PPI对 | 用途 |
|------|---------|----------|------|
| ARATH（拟南芥）| ~5,000 | ~20,000 | 植物泛化 |
| YEAST（酵母） | ~6,000 | ~30,000 | 真菌泛化 |
| ECOLI（大肠杆菌）| ~4,000 | ~15,000 | 细菌泛化 |

## 🔧 数据文件格式

### 1. PPI对文件（.txt）
```
# human_train_ppi.txt
protein1_id protein2_id label
P12345 Q67890 1
P11111 Q22222 0
...
```
- `label=1`: 正样本（真实相互作用）
- `label=0`: 负样本（无相互作用）

### 2. 序列文件（.fasta）
```
>P12345
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPW...
>Q67890
MVSKGEEDNMASLPATHELHIFGSINGVDFDMVGQGTGNPNDGYEELNLKSTKGDL...
```

### 3. 图文件（.pkl）
Python pickle格式，包含NetworkX图对象：
```python
import pickle
import networkx as nx

with open('human_train_graph.pkl', 'rb') as f:
    G = pickle.load(f)
    
# G是NetworkX图对象
nodes = list(G.nodes())
edges = list(G.edges())
```

### 4. 蛋白质ID映射（.csv）
```csv
uniprot_id,organism_code,sequence,sequence_length
P12345,HUMAN,MSKGEE...,238
Q67890,HUMAN,MVSKGE...,267
```

## 🎯 使用场景

### 场景1：种内网络重建（Intra-species）

**目标**：在人类PPI网络上训练，在人类测试集上评估

```python
# 数据路径
train_ppi = "data/PRING/.../human/BFS/human_train_ppi.txt"
val_ppi = "data/PRING/.../human/BFS/human_val_ppi.txt"
test_ppi = "data/PRING/.../human/BFS/all_test_ppi.txt"
sequences = "data/PRING/.../human/human_simple.fasta"

# 训练流程
model = YourModel()
model.train(train_ppi, val_ppi, sequences)

# 推理
predictions = model.predict(test_ppi, sequences)

# 评估（拓扑指标）
python data/PRING/topology_task/eval.py \
    --ppi_path predictions.txt \
    --gt_graph_path data/PRING/.../human/BFS/human_test_graph.pkl \
    --test_graph_node_path data/PRING/.../human/BFS/test_sampled_nodes.pkl
```

**评估指标**：
- Graph Similarity（图相似度）
- Relative Density（相对密度）
- Degree Distribution MMD（度分布）
- Clustering Coefficient MMD（聚类系数）
- Spectral MMD（谱距离）

### 场景2：跨物种泛化（Cross-species）

**目标**：在人类数据上训练，在其他物种上测试泛化能力

```python
# 1. 在人类数据上训练（同场景1）
model.train(human_train, human_val, sequences)

# 2. 在其他物种上测试
for species in ['arath', 'yeast', 'ecoli']:
    test_ppi = f"data/PRING/.../{ species}/{species}_all_test_ppi.txt"
    sequences = f"data/PRING/.../{ species}/{species}_simple.fasta"
    
    predictions = model.predict(test_ppi, sequences)
    
    # 评估
    evaluate(predictions, species)
```

### 场景3：二分类快速迭代

**目标**：快速验证模型，不进行完整图重建

```python
# 使用 human_test_ppi.txt 而不是 all_test_ppi.txt
test_ppi = "data/PRING/.../human/BFS/human_test_ppi.txt"

predictions = model.predict(test_ppi, sequences)

# 计算传统指标
from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(labels, predictions)
aupr = average_precision_score(labels, predictions)
```

## 🔄 三种采样策略

PRING提供三种网络采样策略，模拟不同的网络拓扑：

### 1. BFS（广度优先搜索）
- 特点：从核心节点向外扩展
- 适用：研究中心-外围结构
- 测试图：连通性强，度分布相对均匀

### 2. DFS（深度优先搜索）
- 特点：探索长路径和链式结构
- 适用：研究信号通路
- 测试图：包含更多长距离连接

### 3. Random Walk（随机游走）
- 特点：基于邻域随机性
- 适用：模拟实际发现过程
- 测试图：更接近真实采样

**建议**：
- 主要实验选择一种策略（推荐BFS）
- 消融实验评估在三种策略下的稳定性

## 📈 评估指标详解

### 拓扑指标（Topology Metrics）

**1. Graph Similarity（图相似度）**
- 基于图编辑距离
- 范围：[0, 1]，越高越好
- 衡量整体结构相似性

**2. Relative Density（相对密度）**
```
RD = |edges_pred| / |edges_gt|
```
- 范围：[0, ∞]，理想值=1
- 衡量边数量的准确性

**3. Degree Distribution MMD（度分布）**
- Maximum Mean Discrepancy
- 越小越好
- 衡量节点度分布的相似性

**4. Clustering Coefficient MMD（聚类系数）**
- 衡量局部聚集性
- 越小越好
- 反映三角形结构的保持

**5. Spectral MMD（谱距离）**
- 基于拉普拉斯矩阵特征值
- 越小越好
- 衡量全局结构特性

### 功能指标（Function Metrics）

详见：
- `data/PRING/complex_pathway/` - 蛋白质复合物通路预测
- `data/PRING/enrichment_analysis/` - GO富集分析
- `data/PRING/essential_protein/` - 必需蛋白鉴定

## 🚀 快速开始

### 步骤1：检查数据
```bash
cd data/PRING/data_process/pring_dataset/human/BFS
ls -lh

# 查看文件内容
head human_train_ppi.txt
head -n 4 ../human_simple.fasta
```

### 步骤2：加载数据（示例）
```python
import pandas as pd
from Bio import SeqIO

# 加载PPI对
ppi_df = pd.read_csv(
    'data/PRING/.../human/BFS/human_train_ppi.txt',
    sep=' ',
    names=['protein1', 'protein2', 'label']
)

# 加载序列
sequences = {}
with open('data/PRING/.../human/human_simple.fasta') as f:
    for record in SeqIO.parse(f, 'fasta'):
        sequences[record.id] = str(record.seq)

print(f"训练PPI对: {len(ppi_df)}")
print(f"蛋白质序列: {len(sequences)}")
```

### 步骤3：创建PyTorch Dataset
```python
from torch.utils.data import Dataset

class PRINGDataset(Dataset):
    def __init__(self, ppi_file, fasta_file):
        self.ppi_df = pd.read_csv(ppi_file, sep=' ', 
                                   names=['protein1', 'protein2', 'label'])
        self.sequences = self.load_sequences(fasta_file)
    
    def load_sequences(self, fasta_file):
        sequences = {}
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences[record.id] = str(record.seq)
        return sequences
    
    def __len__(self):
        return len(self.ppi_df)
    
    def __getitem__(self, idx):
        row = self.ppi_df.iloc[idx]
        seq1 = self.sequences[row['protein1']]
        seq2 = self.sequences[row['protein2']]
        label = row['label']
        return seq1, seq2, label
```

## 📚 相关资源

- **论文**: [PRING: Rethinking PPI Prediction from Pairs to Graphs](https://arxiv.org/abs/2507.05101)
- **GitHub**: [https://github.com/SophieSarceau/PRING](https://github.com/SophieSarceau/PRING)
- **评估代码**: `data/PRING/topology_task/eval.py`
- **数据格式**: `data/PRING/data_process/data_format.md`

## 🔍 常见问题

**Q: 需要重新下载PRING数据吗？**
A: 不需要，数据已在 `data/PRING/` 中。

**Q: 应该选择哪种采样策略？**
A: 推荐从BFS开始，它的结果最稳定。

**Q: 如何处理序列ID？**
A: PRING使用UniProt ID，格式标准，可直接用于ESM等模型。

**Q: 负样本是如何采样的？**
A: PRING使用拓扑驱动策略，避免了简单随机采样的问题。

**Q: 可以只使用human数据吗？**
A: 可以，跨物种测试是可选的，用于评估泛化能力。

## 🎓 引用

如果使用PRING数据集，请引用：

```bibtex
@inproceedings{zheng2025pring,
  title={{PRING}: Rethinking Protein-Protein Interaction Prediction from Pairs to Graphs},
  author={Xinzhe Zheng and Hao Du and Fanding Xu and Jinzhe Li and Zhiyuan Liu and Wenkang Wang and Tao Chen and Wanli Ouyang and Stan Z. Li and Yan Lu and Nanqing Dong and Yang Zhang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```

---

**最后更新**: 2025年10月4日  
**维护者**: Hyperbolic_Big项目组

