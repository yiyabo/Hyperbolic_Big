# 数据预处理模块使用指南

## 📋 概述

数据预处理模块为层次化特征建模提供完整的数据过滤和准备功能，支持：

- ✅ **PPI网络过滤**: 基于置信度和蛋白质质量的多重过滤
- ✅ **蛋白质质量评估**: 基于长度、注释、物种等的综合质量评分
- ✅ **MoE专家分组**: 基于STRING聚类的专家模型分组
- ✅ **图连通性分析**: 确保过滤后的网络连通性
- ✅ **统计分析和可视化**: 全面的数据统计和图表生成

## 🏗️ 模块结构

```
data_preprocessing/
├── __init__.py                 # 模块入口
├── ppi_filter.py              # PPI数据过滤器
├── protein_filter.py          # 蛋白质质量过滤器
├── cluster_analyzer.py        # 聚类分析器
└── data_statistics.py         # 数据统计分析器

run_data_preprocessing.py       # 主运行脚本
test_data_preprocessing.py      # 测试脚本
```

## 🚀 快速开始

### 1. 运行完整的预处理流程

```bash
# 使用默认参数（推荐）
python run_data_preprocessing.py

# 自定义参数
python run_data_preprocessing.py \
    --confidence 0.7 \
    --quality-threshold 0.5 \
    --max-experts 50 \
    --min-cluster-size 100
```

### 2. 测试功能

```bash
# 运行功能测试（使用小样本）
python test_data_preprocessing.py
```

## ⚙️ 主要参数说明

### 过滤参数
- `--confidence`: PPI置信度阈值 (0-1, 默认0.7)
- `--quality-threshold`: 蛋白质质量阈值 (0-1, 默认0.5)

### MoE专家参数
- `--max-experts`: 最大专家组数量 (默认50)
- `--min-cluster-size`: 专家组最小聚类大小 (默认100)
- `--use-hierarchical`: 使用层次化专家分组
- `--hierarchical-depth`: 层次聚类深度 (默认2)

### 输出控制
- `--data-dir`: 输入数据目录 (默认data)
- `--output-dir`: 输出目录 (默认data/filtered)
- `--analysis-dir`: 分析结果目录 (默认analysis_results)
- `--skip-plots`: 跳过绘图

## 📊 输出文件说明

### 主要数据文件
```
data/filtered/
├── filtered_proteins_conf0.7.csv      # 过滤后的蛋白质数据
├── filtered_ppi_conf0.7.csv           # 过滤后的PPI数据
├── expert_groups_conf0.7.json         # MoE专家分组
└── filtering_stats_conf0.7.json       # 过滤统计信息
```

### 分析结果
```
analysis_results/
├── comprehensive_data_report.json     # 综合数据报告
├── ppi_network_statistics.png         # PPI网络统计图表
└── protein_statistics.png             # 蛋白质统计图表
```

## 🔧 高级使用

### 1. 单独使用各模块

```python
from data_preprocessing import PPIDataFilter, ClusterAnalyzer, DataStatistics

# PPI数据过滤
ppi_filter = PPIDataFilter(confidence_threshold=0.7)
proteins_df, ppi_df, files = ppi_filter.run_complete_filtering()

# 聚类分析
cluster_analyzer = ClusterAnalyzer()
expert_groups = cluster_analyzer.get_moe_expert_groups()

# 统计分析
stats_analyzer = DataStatistics()
report = stats_analyzer.generate_comprehensive_report(proteins_df, ppi_df)
```

### 2. 自定义过滤策略

```python
# 自定义蛋白质质量过滤
from data_preprocessing.protein_filter import ProteinQualityFilter

quality_filter = ProteinQualityFilter()
# 修改过滤参数
quality_filter.length_limits['min_length'] = 30
quality_filter.length_limits['max_length'] = 3000

filtered_proteins, stats = quality_filter.filter_proteins(protein_df)
```

### 3. 不同的专家分组策略

```python
# 平铺聚类策略
expert_groups = cluster_analyzer.get_moe_expert_groups(
    min_cluster_size=200,
    max_num_experts=30
)

# 层次化策略
hierarchical_groups = cluster_analyzer.get_hierarchical_expert_groups(
    target_depth=3
)

# 随机分组策略（备选）
random_groups = cluster_analyzer._create_random_expert_groups(
    num_experts=20,
    protein_list=list(proteins_df['protein_id'])
)
```

## 📈 数据质量控制

### 蛋白质质量评分标准

- **长度分数** (30%权重):
  - 最佳长度: 100-1000 氨基酸 → 1.0分
  - 可接受长度: 50-5000 氨基酸 → 0.5-1.0分
  - 异常长度: <50 或 >5000 → 0.0分

- **注释分数** (40%权重):
  - 高质量关键词: +0.2分 (characterized, enzyme等)
  - 低质量关键词: -0.15分 (hypothetical, fragment等)
  - 预测方法: -0.05分 (computational analysis等)

- **名称分数** (20%权重):
  - 标准基因名: +0.2分 (acsA, hsp70等)
  - 通用ID: -0.1分 (ABC123.1等)

- **物种分数** (10%权重):
  - 模式生物: 0.8-1.0分
  - 其他物种: 0.6分

### PPI过滤策略

1. **置信度过滤**: combined_score ≥ threshold × 1000
2. **蛋白质质量过滤**: 只保留质量评分合格的蛋白质
3. **图连通性过滤**: 保留最大连通分量
4. **多通道证据分析**: 分析8个证据通道的贡献

## 🔍 故障排除

### 常见问题

**Q: 聚类文件不存在怎么办？**
A: 系统会自动降级使用随机分组策略，不影响主流程。

**Q: 内存不足怎么办？**
A: 可以增加chunk_size参数或在代码中调整批处理大小。

**Q: 过滤后数据太少怎么办？**
A: 降低置信度阈值（如0.6）或质量阈值（如0.4）。

**Q: 图不连通怎么办？**
A: 系统会自动保留最大连通分量，小的连通分量会被过滤。

### 性能优化

- **并行处理**: 可以修改代码支持多进程处理
- **内存优化**: 使用chunking策略处理大文件
- **存储优化**: 使用HDF5或Parquet格式存储大数据

## 📋 预期输出示例

```
================================================================================
                      数据处理摘要报告
================================================================================

📊 基础数据统计:
   蛋白质总数: 1,234,567
   相互作用总数: 5,678,901
   网络中蛋白质数: 987,654
   平均度数: 11.49

🧬 蛋白质属性:
   平均长度: 345.2 ± 198.7
   长度范围: 50 - 4999
   物种数量: 4,521
   平均质量分数: 0.634
   高质量蛋白质: 456,789 (37.0%)

🔗 PPI网络属性:
   平均置信度: 823.4
   置信度范围: 700 - 999
   证据通道使用率:
     experimental: 23.4%
     database: 45.6%
     textmining: 78.9%
     coexpression: 34.2%

🎭 MoE专家组:
   专家组数量: 42
   覆盖蛋白质: 876,543
   平均组大小: 20,870.5 ± 15,432.1
   组大小范围: 123 - 89,456

📈 数据过滤效果:
   蛋白质保留率: 67.8%
   PPI保留率: 23.4%

================================================================================
✅ 数据预处理完成，可以开始PSSM生成！
================================================================================
```

## 🔗 下一步

数据预处理完成后，您可以：

1. **开始PSSM生成**: 使用过滤后的蛋白质序列
2. **设计小波变换**: 基于PSSM矩阵
3. **训练MoE模型**: 使用专家分组信息
4. **构建HGCN**: 使用过滤后的PPI网络

预处理的输出数据已经针对您的层次化特征建模框架进行了优化！