# 蛋白质相互作用预测：层次化特征建模框架

## 项目概述

本项目实现了一个用于蛋白质相互作用预测的层次化特征工程与建模框架，通过多步骤、系统化的特征工程，主动地为每个蛋白质构建蕴含多重层次结构的特征表征，然后输入到双曲空间模型（HGCN）中，实现特征与模型的协同效应。

## 核心思想

> **通过显式地构建层次化特征，与双曲空间模型的天然层次表达能力相匹配，实现 1+1 > 2 的协同效应。**

## 四步骤建模流程

1. **进化信息编码 (PSSM)** - 使用PSI-BLAST获取蛋白质序列的进化保守性信息
2. **多尺度层次提取 (小波变换)** - 对PSSM矩阵应用2D小波变换，提取不同尺度的保守性模式  
3. **专家化模型 (MoE)** - 引入混合专家系统，针对不同蛋白质家族进行专门化处理
4. **双曲空间图学习 (HGCN)** - 在双曲空间中学习整个PPI网络的拓扑结构

## 数据来源

- **STRING数据库 v12.0** - 使用置信度 > 0.95 的高质量蛋白质相互作用数据
- **跨物种数据** - 利用多物种信息增强模型的泛化能力

## 环境配置

### 依赖安装

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\\Scripts\\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖包

- **数据处理**: pandas, numpy, scipy
- **生物信息学**: biopython
- **机器学习**: torch, torch-geometric, scikit-learn  
- **信号处理**: pywt (小波变换)
- **网络请求**: requests
- **进度显示**: tqdm

## 使用指南

### 1. 数据提取

首先从STRING数据库提取高置信度的蛋白质相互作用数据：

```bash
cd data_extraction
python string_data_extractor.py
```

这个脚本将：
- 下载STRING v12.0的完整数据集
- 筛选置信度 > 0.95 的相互作用
- 提取蛋白质序列和注释信息
- 将数据存储到SQLite数据库中

**输出文件**:
- `data/string_data.db` - 包含所有数据的SQLite数据库
- `data/` - 原始下载文件的缓存目录

### 2. 进化信息编码（即将实现）

使用PSI-BLAST生成PSSM矩阵：

```bash
# 即将实现
python pssm_generator.py
```

### 3. 小波变换特征提取（即将实现）

```bash
# 即将实现  
python wavelet_feature_extractor.py
```

### 4. 混合专家系统（即将实现）

```bash
# 即将实现
python moe_system.py
```

### 5. HGCN模型训练（即将实现）

```bash
# 即将实现
python train_hgcn.py
```

## 数据统计示例

提取完成后，你将看到类似如下的统计信息：

```
STRING数据提取完成！
==================================================
总蛋白质数量: 24,584,628
总物种数量: 5,090
高置信度相互作用: 8,926,434
有序列的蛋白质: 24,584,628

前10个物种的蛋白质数量:
  物种 511145: 4,391 个蛋白质  # 大肠杆菌
  物种 9606: 19,614 个蛋白质   # 人类
  物种 83333: 4,391 个蛋白质   # 大肠杆菌 K-12
  ...
```

## 项目结构

```
Hyperbolic_Big/
├── docs/                              # 文档
│   ├── hierarchical_feature_modeling_strategy.md
│   ├── implementation_strategy.md  
│   └── downstream_tasks_analysis.md
├── data_extraction/                   # 数据提取模块
│   └── string_data_extractor.py
├── pssm_generation/                   # PSSM生成模块（即将实现）
├── wavelet_features/                  # 小波特征提取模块（即将实现）  
├── moe_system/                        # 混合专家系统模块（即将实现）
├── hgcn_model/                        # HGCN模型模块（即将实现）
├── requirements.txt                   # 依赖文件
└── README.md                          # 本文件
```

## 预期优势

1. **性能提升** - 特征的内在层次性与模型的几何层次性完美匹配
2. **可解释性增强** - 每个步骤都有明确的生物学意义
3. **生物学先验融入** - 基于明确的生物学假设，具有更好的泛化能力

## 注意事项

- **计算资源需求较大** - PSSM生成和大规模数据处理需要充足的计算资源
- **存储空间需求** - STRING完整数据集较大，需要足够的存储空间
- **网络连接** - 数据下载过程需要稳定的网络连接

## 下一步计划

1. ✅ 完成STRING数据提取
2. 🔄 实现PSSM生成模块  
3. ⏳ 实现小波变换特征提取
4. ⏳ 构建混合专家系统
5. ⏳ 集成HGCN模型

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

[待定]
