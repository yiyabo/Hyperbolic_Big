# STRING数据库获取策略：支持层次化特征建模

**版本**: 1.0  
**日期**: 2025年9月22日  
**基于理论**: `hierarchical_feature_modeling_strategy.md`

---

## 📋 总结

基于你的层次化特征建模理论，我们需要的不是简单的STRING基础数据，而是一个**完整的、支持多层次特征工程的数据集合**。

## 🎯 理论需求映射

### 你的四步理论框架 → 对应数据需求

| 建模步骤 | 理论组件 | 所需数据集 | 状态 |
|---------|---------|-----------|------|
| **第一步** | PSSM生成（进化信息编码） | `protein.sequences.v12.0.fa.gz` | ✅ 已包含 |
| **第二步** | 小波变换（多尺度层次提取） | PSSM矩阵（基于序列生成） | ✅ 可实现 |
| **第三步** | MoE专家模型（蛋白质家族专门化） | `clusters.*.v12.0.txt.gz` | ✅ 新增 |
| **第四步** | HGCN图学习（双曲空间建模） | `protein.links.detailed.v12.0.txt.gz` | ✅ 新增 |

## 🗂️ 完整数据集清单

### 基础数据（原有）
- **protein.info.v12.0.txt.gz**: 蛋白质基本信息
- **protein.sequences.v12.0.fa.gz**: 蛋白质序列（用于PSSM生成）
- **protein.links.v12.0.txt.gz**: 基础相互作用数据

### 层次化建模增强数据（新增）
- **protein.links.detailed.v12.0.txt.gz**: 多通道证据评分
  - 包含8个证据通道：neighborhood, fusion, cooccurrence, coexpression, experimental, database, textmining
  - 为HGCN提供更丰富的图结构信息

- **clusters.info.v12.0.txt.gz**: 聚类信息
  - 聚类ID、名称、描述、大小
  - 为MoE专家模型提供家族分类基础

- **clusters.proteins.v12.0.txt.gz**: 蛋白质聚类映射
  - 每个蛋白质所属的聚类信息
  - 直接支持MoE的专家路由

- **clusters.tree.v12.0.txt.gz**: 聚类层次树
  - 父子聚类关系和距离
  - 支持层次化的MoE架构

## 🏗️ 数据库架构

我们的SQLite数据库现在包含以下表结构：

```sql
-- 基础表
protein_info              -- 蛋白质基本信息
protein_sequences         -- 蛋白质序列（用于PSSM）
protein_interactions      -- 基础相互作用

-- 层次化建模增强表
protein_interactions_detailed  -- 多通道详细相互作用
cluster_info                   -- 聚类信息（MoE专家分类）
protein_clusters              -- 蛋白质-聚类映射
cluster_tree                  -- 聚类层次树
```

## 🔧 实施优势

### 相对于原方案的改进：
1. **MoE专家模型支持**: 通过STRING聚类数据，我们可以实现基于生物学意义的专家分工
2. **多通道图信息**: 详细的证据评分允许HGCN学习更丰富的图结构
3. **层次信息显式化**: 聚类树提供了明确的层次关系，与双曲几何高度契合
4. **数据一致性**: 所有数据来自同一个STRING版本，确保一致性

## ⚠️ 注意事项与限制

### 关于蛋白质家族分类：
- **STRING聚类** vs **Pfam分类**：STRING的聚类是基于序列相似性和功能关联的，虽然不如Pfam精确，但对MoE建模仍然有效
- **可选增强**：如果需要更精确的家族信息，可以考虑集成UniProt/Pfam数据

### 数据量考虑：
- 完整的STRING v12.0数据集非常大（数十GB）
- 建议根据研究需求选择特定物种或置信度阈值进行筛选

## 🚀 下一步建议

1. **运行更新后的数据提取器**，获取完整的层次化建模数据
2. **验证聚类质量**，确保STRING聚类适合作为MoE专家分组
3. **开发PSSM生成管道**，基于序列数据生成小波变换的输入
4. **设计MoE路由策略**，利用聚类信息实现专家分工

---

**结论**: 通过这个扩展的数据获取策略，我们现在有了完整支持你的层次化特征建模理论的数据基础，每个组件都有对应的数据支撑。
