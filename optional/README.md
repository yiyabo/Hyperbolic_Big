# 可选功能模块

本文件夹包含**可选的扩展功能**，不是项目核心流程的必需部分。

## 📦 包含的模块

### 1. data_extraction/
**功能**: STRING数据库数据提取  
**用途**: 可选的大规模预训练或数据增强  
**状态**: 完整实现，可直接使用  

从STRING v12.0提取蛋白质相互作用数据，包括：
- 蛋白质信息和序列
- 高置信度相互作用
- 聚类信息（用于MoE/LoRA）
- 多通道证据评分

**使用方法**:
```bash
cd optional/data_extraction
python string_data_extractor.py
```

### 2. data_preprocessing/
**功能**: STRING数据预处理  
**用途**: 对STRING数据进行过滤、质量控制和统计分析  
**状态**: 完整实现，可直接使用  

提供的功能：
- PPI网络过滤（置信度、质量）
- 图连通性分析
- 聚类分析和专家分组
- 数据统计和可视化

**使用方法**:
```bash
cd optional
python -c "from data_preprocessing import PPIDataFilter; ..."
```

## ⚠️ 注意事项

1. **主数据集**: 项目主要使用 **PRING基准测试数据集**（位于 `data/PRING/`）
2. **可选性**: 这些模块是**可选的扩展**，不影响主流程
3. **资源需求**: STRING完整数据集较大（~50GB），需要足够的存储空间和计算资源
4. **使用场景**:
   - 大规模预训练（迁移学习到PRING）
   - 数据增强（扩充训练集）
   - 额外物种评估（超出PRING的4个物种）

## 📚 相关文档

- [PRING数据集使用指南](../docs/pring_dataset.md) - 主数据集文档
- [STRING数据获取策略](../docs/data_acquisition_strategy.md) - STRING扩展方案
- [数据预处理README](./data_preprocessing/README.md) - 预处理详细说明

---

**推荐**: 如果你只是想快速开始模型开发，**无需使用此文件夹中的功能**。直接使用PRING数据集即可。

