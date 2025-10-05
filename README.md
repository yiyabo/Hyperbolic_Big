# 蛋白质相互作用预测：PLM + RAG + 图小波 × 洛伦兹 GNN 框架

## 项目概述

本项目实现了一个用于蛋白质相互作用预测的现代化深度学习框架，以**蛋白质语言模型（PLM）为主通道**，结合**检索增强生成（RAG）**、**图小波多尺度分析**、**洛伦兹几何神经网络（可学习曲率）**和**家族条件LoRA**，实现高性能、可解释、工程可落地的PPI预测。

## 核心思想

> **在主干上"去MSA压力"（PLM + RAG），在几何/尺度上"强耦合"（图小波 × 洛伦兹GNN），用家族条件LoRA实现轻量专才化，达到性能、可解释性与工程效率的最佳平衡。**

## 五步骤建模流程

1. **PLM主通道 (ESM-3)** - 序列-结构-功能统一表征，替代昂贵的PSSM
2. **检索增强 (RAG-PLM)** - 用MMseqs2轻量检索同源序列，注入共进化信息
3. **家族条件 LoRA** - 参数高效的专才化，替代重型MoE系统
4. **图小波 (GWNN/SGWT)** - 显式编码多尺度邻域/分辨率信息
5. **洛伦兹 GNN (可学习曲率)** - 在双曲空间中自适应学习层次拓扑结构

## 数据来源

**主要数据集：PRING基准测试** (NeurIPS 2025)
- 标准化的PPI预测评估数据集
- 4个精选物种：human（训练）, arath, yeast, ecoli（跨物种测试）
- 已完成高质量预处理（序列过滤、负样本采样、图切分）
- 多种网络拓扑采样策略：BFS, DFS, Random Walk
- 拓扑+功能双重评估任务
- 论文：[PRING: Rethinking PPI Prediction from Pairs to Graphs](https://arxiv.org/abs/2507.05101)

**扩展数据（可选）：**
- **STRING数据库 v12.0** - 可用于大规模预训练或数据增强
- 24M+ 蛋白质，5,090 物种，多通道证据评分

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
- **深度学习**: torch, torch-geometric, transformers
- **蛋白质语言模型**: esm (ESM-2/ESM-3)
- **同源检索**: mmseqs2（需单独安装）
- **图小波**: pygsp 或自定义实现
- **双曲几何**: geoopt（可选）
- **网络请求**: requests
- **进度显示**: tqdm

## 使用指南

### 1. 数据准备

**PRING数据集已就绪**，位于 `data/PRING/data_process/pring_dataset/`

数据集结构：
```
pring_dataset/
├── human/              # 人类（训练集）
│   ├── BFS/           # 广度优先采样切分
│   │   ├── human_train_ppi.txt
│   │   ├── human_val_ppi.txt
│   │   ├── human_test_ppi.txt
│   │   └── all_test_ppi.txt
│   ├── DFS/           # 深度优先采样切分
│   ├── RANDOM_WALK/   # 随机游走采样切分
│   └── human_simple.fasta
├── arath/             # 拟南芥（跨物种测试）
├── yeast/             # 酵母（跨物种测试）
└── ecoli/             # 大肠杆菌（跨物种测试）
```

**可选：STRING数据提取**（用于扩展训练数据）
```bash
cd data_extraction
python string_data_extractor.py
```

### 2. ESM-3特征提取（即将实现）

提取PLM序列表征：

```bash
# 即将实现
python extract_esm_features.py
```

### 3. RAG同源检索（即将实现）

使用MMseqs2检索同源序列：

```bash
# 即将实现
python rag_retrieval.py
```

### 4. 图小波特征（即将实现）

在PPI图上应用多尺度小波变换：

```bash
# 即将实现
python graph_wavelet_features.py
```

### 5. 洛伦兹GNN训练（即将实现）

训练带可学习曲率的双曲空间GNN：

```bash
# 即将实现
python train_lorentz_gnn.py
```

## PRING数据集统计

**Human (训练集)**
- 蛋白质数量: ~19,000
- 训练集PPI对: ~100,000+
- 验证集PPI对: ~10,000+
- 测试集PPI对: ~10,000+
- 三种采样策略: BFS, DFS, Random Walk

**跨物种测试集**
- ARATH (拟南芥): ~5,000 蛋白质
- YEAST (酵母): ~6,000 蛋白质  
- ECOLI (大肠杆菌): ~4,000 蛋白质

**质量保证**
- 序列长度: 50-1000 氨基酸
- 序列相似度过滤: <0.4 (MMseqs2)
- 负样本采样: 拓扑驱动策略
- 功能相似性过滤: 已完成

## 项目结构

```
Hyperbolic_Big/
├── 📚 docs/                           # 文档
│   ├── new_method.md                  # ⭐ 新方法核心（PLM+RAG+图小波+洛伦兹GNN）
│   ├── plm_lora.md                    # 家族条件LoRA设计规范
│   ├── pring_dataset.md               # PRING数据集使用指南
│   ├── implementation_strategy.md     # HGCN数学推导与实现策略
│   ├── data_acquisition_strategy.md   # 数据获取策略（STRING，可选）
│   ├── downstream_tasks_analysis.md   # 下游任务分析
│   └── old/                           # 旧版方法归档
│
├── 📦 data/                           # 数据
│   └── PRING/                         # ⭐ PRING基准测试数据集（NeurIPS 2025）
│       ├── data_process/pring_dataset/    # 主数据集
│       │   ├── human/                     # 人类（训练）
│       │   ├── arath/                     # 拟南芥（测试）
│       │   ├── yeast/                     # 酵母（测试）
│       │   └── ecoli/                     # 大肠杆菌（测试）
│       ├── topology_task/                 # 拓扑任务评估
│       ├── complex_pathway/               # 复合物通路评估
│       ├── enrichment_analysis/           # GO富集分析
│       └── essential_protein/             # 必需蛋白鉴定
│
├── ✅ data_loader/                    # PRING数据加载器（已实现）
│   ├── __init__.py
│   ├── config.py                      # 配置管理
│   ├── pring_dataset.py               # Dataset类
│   └── README.md                      # 使用文档
│
├── 📝 examples/                       # 使用示例
│   └── load_pring_data.py             # 数据加载示例
│
├── 🔄 esm_features/                   # ESM特征提取（即将实现）
├── 🔄 rag_retrieval/                  # RAG检索（即将实现）
├── 🔄 graph_wavelet/                  # 图小波（即将实现）
├── 🔄 family_lora/                    # 家族条件LoRA（即将实现）
├── 🔄 lorentz_gnn/                    # 洛伦兹GNN（即将实现）
│   ├── geometry/                      # 几何操作
│   ├── layers/                        # GNN层
│   └── models/                        # 完整模型
├── 🔄 evaluation/                     # 评估模块（即将实现）
│
├── 📦 optional/                       # 可选扩展功能
│   ├── data_extraction/               # STRING数据提取（可选）
│   ├── data_preprocessing/            # STRING数据预处理（可选）
│   └── README.md                      # 说明文档
│
├── requirements.txt                   # Python依赖
├── test_data_loader.py                # 数据加载器测试
└── README.md                          # 本文件
```

## 预期优势

1. **性能与可扩展性** - PLM主通道 + RAG注入共进化，较纯单序列更强，同时避免全量MSA昂贵代价
2. **几何契合** - 图小波显式"尺度"，洛伦兹GNN承载"层级"，二者互补
3. **工程友好** - 家族条件LoRA取代大MoE，参数效率与稳定性更佳
4. **可解释性** - 同源检索权重 → 图小波尺度能量 → 洛伦兹半径/角坐标分布
5. **成本控制** - 用MMseqs2（GPU加速）替代PSI-BLAST，用LoRA替代MoE

## 注意事项

- **ESM模型规模** - ESM-3模型较大，建议使用GPU进行特征提取
- **存储空间需求** - STRING完整数据集较大，需要足够的存储空间（~50GB）
- **网络连接** - 数据下载和模型下载需要稳定的网络连接
- **InterPro版本** - 固定InterProScan/InterPro版本号以保证可复现性
- **数值稳定性** - 洛伦兹几何操作需要注意梯度裁剪和数值稳定性

## 下一步计划

### 阶段一：数据与特征提取 ✅
1. ✅ PRING基准测试数据集已就绪
2. 🔄 实现PRING数据加载器（PyTorch Dataset）
3. 🔄 实现ESM-3特征提取模块

### 阶段二：核心模型组件
4. ⏳ 实现RAG同源检索（MMseqs2集成）
5. ⏳ 实现图小波多尺度特征（GWNN/SGWT）
6. ⏳ 实现家族条件LoRA（InterPro/Pfam标注）
7. ⏳ 实现洛伦兹几何核心（Lorentz流形操作）
8. ⏳ 实现洛伦兹GNN层（可学习曲率）

### 阶段三：训练与评估
9. ⏳ 完整训练流程（Human数据，三种采样策略）
10. ⏳ 拓扑任务评估（种内+跨物种）
11. ⏳ 功能任务评估（复合物、GO富集、必需蛋白）
12. ⏳ 消融实验（PLM vs PSSM, LoRA vs MoE, 图小波等）
13. ⏳ 与PRING基线方法对比

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

[待定]
