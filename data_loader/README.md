# PRING数据加载器使用指南

## 📋 概述

提供标准化的PyTorch Dataset类，用于加载和处理PRING基准测试数据。

## 🚀 快速开始

### 基本使用

```python
from data_loader import PRINGPairDataset, PRINGConfig, get_dataloader

# 1. 创建配置
config = PRINGConfig(
    species="human",
    sampling_strategy="BFS",
    split="train"
)

# 2. 创建数据集
dataset = PRINGPairDataset(config)

# 3. 创建DataLoader
dataloader = get_dataloader(dataset, batch_size=32, shuffle=True)

# 4. 遍历数据
for batch in dataloader:
    seq1 = batch['seq1']  # List[str]
    seq2 = batch['seq2']  # List[str]
    labels = batch['label']  # torch.Tensor
    
    # 你的训练代码...
```

### 数据集统计

```python
# 查看数据集统计信息
stats = dataset.get_statistics()
print(f"总PPI对: {stats['num_pairs']}")
print(f"蛋白质数: {stats['num_proteins']}")
print(f"正样本比例: {stats['positive_ratio']:.2%}")
print(f"平均序列长度: {stats['avg_seq_length']:.1f}")
```

## 🎯 支持的配置

### 物种 (species)
- `"human"` - 人类（训练集）
- `"arath"` - 拟南芥（测试集）
- `"yeast"` - 酵母（测试集）
- `"ecoli"` - 大肠杆菌（测试集）

### 采样策略 (sampling_strategy)
仅对human有效：
- `"BFS"` - 广度优先搜索（推荐）
- `"DFS"` - 深度优先搜索
- `"RANDOM_WALK"` - 随机游走

### 数据切分 (split)
- `"train"` - 训练集（仅human）
- `"val"` - 验证集（仅human）
- `"test"` - 测试集（二分类评估）
- `"all_test"` - 完整测试集（图重建评估）

## 📂 数据路径配置

### 方式1：使用默认路径

```python
# 自动使用项目中的data/PRING/...
config = PRINGConfig(species="human", split="train")
```

### 方式2：设置环境变量

```bash
# 在~/.bashrc 或 ~/.zshrc 中添加
export PRING_DATA_ROOT="/path/to/your/PRING/data_process/pring_dataset"
```

```python
# 代码中会自动读取环境变量
config = PRINGConfig(species="human", split="train")
```

### 方式3：显式指定路径

```python
config = PRINGConfig(
    data_root="/custom/path/to/pring_dataset",
    species="human",
    split="train"
)
```

## 💡 使用场景

### 场景1：训练模型（人类数据）

```python
from data_loader import PRINGPairDataset, PRINGConfig, get_dataloader

# 训练集
train_config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
train_dataset = PRINGPairDataset(train_config)
train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)

# 验证集
val_config = PRINGConfig(species="human", sampling_strategy="BFS", split="val")
val_dataset = PRINGPairDataset(val_config)
val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False)

# 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        # 训练代码...
        pass
    
    # 验证
    for batch in val_loader:
        # 验证代码...
        pass
```

### 场景2：快速评估（二分类）

```python
# 测试集
test_config = PRINGConfig(species="human", sampling_strategy="BFS", split="test")
test_dataset = PRINGPairDataset(test_config, return_ids=True)
test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

# 评估
all_preds = []
all_labels = []

for batch in test_loader:
    preds = model.predict(batch['seq1'], batch['seq2'])
    all_preds.extend(preds)
    all_labels.extend(batch['label'])

# 计算指标
from sklearn.metrics import roc_auc_score, average_precision_score
auc = roc_auc_score(all_labels, all_preds)
aupr = average_precision_score(all_labels, all_preds)
```

### 场景3：图重建评估（完整PRING评估）

```python
from data_loader import PRINGGraphDataset

# 使用all_test数据
config = PRINGConfig(species="human", sampling_strategy="BFS", split="all_test")
dataset = PRINGGraphDataset(config, load_graph=True)
loader = get_dataloader(dataset, batch_size=64, shuffle=False)

# 预测所有边
predictions = []
for batch in loader:
    preds = model.predict(batch['seq1'], batch['seq2'])
    for i in range(len(preds)):
        predictions.append((
            batch['protein1_id'][i],
            batch['protein2_id'][i],
            int(preds[i] > 0.5)  # 二值化
        ))

# 保存预测结果
dataset.save_predictions(predictions, "human_BFS_all_test_ppi_pred.txt")

# 运行PRING评估脚本
import subprocess
subprocess.run([
    "python", "data/PRING/topology_task/eval.py",
    "--ppi_path", "human_BFS_all_test_ppi_pred.txt",
    "--gt_graph_path", str(config.test_graph_file),
    "--test_graph_node_path", str(config.sampled_nodes_file)
])
```

### 场景4：跨物种泛化测试

```python
# 在人类数据上训练（场景1）
# ...

# 在其他物种上测试
for species in ['arath', 'yeast', 'ecoli']:
    test_config = PRINGConfig(species=species, split="all_test")
    test_dataset = PRINGGraphDataset(test_config)
    test_loader = get_dataloader(test_dataset, batch_size=64)
    
    # 预测和评估
    # ...
```

## 🔧 自定义序列转换

```python
def tokenize_sequence(seq: str) -> torch.Tensor:
    """自定义序列转换函数"""
    # 例如：使用ESM tokenizer
    from transformers import EsmTokenizer
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    return tokenizer(seq, return_tensors="pt")['input_ids']

# 在数据集中使用
dataset = PRINGPairDataset(config, transform=tokenize_sequence)
```

## 📊 数据集类

### PRINGPairDataset
用于成对PPI预测（二分类任务）

**特点**：
- 返回序列对和标签
- 支持自定义转换函数
- 内置序列长度过滤
- 提供数据统计

**返回格式**：
```python
{
    'seq1': str,          # 第一个蛋白质序列
    'seq2': str,          # 第二个蛋白质序列
    'label': int,         # 标签 (0或1)
    'protein1_id': str,   # ID (可选)
    'protein2_id': str    # ID (可选)
}
```

### PRINGGraphDataset
用于图重建任务

**特点**：
- 加载all-against-all测试对
- 自动加载真实图结构
- 提供预测结果保存功能
- 支持PRING评估脚本

**额外方法**：
- `get_all_proteins()` - 获取所有蛋白质ID
- `save_predictions()` - 保存预测结果为PRING格式

## 🔍 预定义配置

快速使用常见配置：

```python
from data_loader.config import (
    HUMAN_TRAIN_BFS,
    HUMAN_VAL_BFS,
    HUMAN_TEST_BFS,
    ARATH_TEST,
    YEAST_TEST,
    ECOLI_TEST
)

# 直接使用
train_dataset = PRINGPairDataset(HUMAN_TRAIN_BFS)
```

## 🐛 故障排除

### 问题1：找不到数据文件

```python
# 检查配置
config = PRINGConfig(species="human", split="train")
print(config)

# 验证文件
if not config.validate():
    print("数据文件缺失，请检查路径")
```

### 问题2：序列缺失

数据加载器会自动过滤缺少序列的PPI对，并显示警告。

### 问题3：路径问题（本地 vs 服务器）

```python
# 方法1：环境变量（推荐）
# 服务器上设置：export PRING_DATA_ROOT="/data/pring/..."

# 方法2：代码中动态判断
import os
if os.path.exists("/server/data/PRING"):
    data_root = "/server/data/PRING/data_process/pring_dataset"
else:
    data_root = None  # 使用默认路径
```

## 📈 性能建议

- **num_workers**: 建议设置为4-8，根据CPU核心数调整
- **batch_size**: 根据GPU显存调整，ESM-2推荐32-64
- **序列长度**: 默认限制1000，可根据需要调整

## 📚 相关文档

- [PRING数据集完整文档](../docs/pring_dataset.md)
- [PRING官方仓库](https://github.com/SophieSarceau/PRING)
- [PRING论文](https://arxiv.org/abs/2507.05101)

