# 方案构想：用于蛋白质相互作用预测的 **PLM + RAG + 图小波 × 洛伦兹 GNN（可学习曲率）** 与 **家族条件 Adapter/LoRA** 框架

**版本**: 2.0
**日期**: 2025年9月24日
**作者**：Yiyabo

---

## 1. 核心思想与目标

之前的项目是“序列驱动的 PPI 预测正在从“重对齐/重进化特征（MSA/PSSM）””，因为各种各样的原因（开会再说）转向“**大规模蛋白语言模型（PLM）** + 轻量注入进化知识”的框架。我们提出的框架以 **PLM 表征为主通道**，通过**检索增强（RAG-PLM）**注入同源/共进化线索，并在 **PPI 图上做多尺度图小波**后，置入**洛伦兹模型的超曲率 GNN（曲率可学习）**进行层级拓扑建模；以 **家族条件 Adapter/LoRA** 替代重型 MoE，最后通过**（可选）PSSM 作为特权信息/解释支路**，只在训练或评测阶段使用，以证明稳健性与可解释价值。
目标：**提升性能**、**增强可解释性**、**降低算力与工程成本**，并显式对齐“**尺度—家族—几何**”。

支撑动因（近年共识与证据）：

* **ESM-3** 将**序列-结构-功能**统一到同一 PLM （既然我们之前用结构数据比较困难，那么就采用ESM-3，也算是融入了结构数据）([BioRxiv][1])
* **RAG-PLM**/RAGFold 等工作证实：用**检索到的 MSA/同源**增强 PLM，在困惑度、contact、fitness 等指标上优于纯单序列 PLM。([BioRxiv][2])
* **图小波（GWNN/SGWT）** 提供图                上多尺度可解释滤波，且已被多篇图学习工作验证高效。([arXiv][3])
* **超曲率/洛伦兹模型 GNN**更贴合层级/幂律拓扑，并可**学习曲率**以自适应数据结构。([CVF开放获取][4])
* PLM 表征已在多项 PPI/界面任务中**替代 PSSM**而不降分，甚至更优。([Oxford Academic][5])

---

## 2. 核心假设

1. PPI 的可预测性源于**同源/共进化**、**结构/模体**与**家族**多层次先验；
2. 这些先验具有**层次与尺度性**，应在模型前向中被**显式编码**（图小波、多曲率）；
3. 在**双曲/洛伦兹**几何中组织这些层次信息，可降低失真并提升泛化。

---

## 3. 实施流程（MSA-free 主线 + 轻量进化注入 + 图几何）

### 第一步：PLM 主通道（MSA-free）

* **输入**：蛋白序列；**特征**：ESM-3（或 ESM-2）残基/序列级向量（可做残基池化/CLS）。
* **理由**：ESM-3 在**序列-结构-功能**三模态联合训练，适合通用下游；资源受限可先用 ESM-2。([BioRxiv][1])

### 第二步：检索增强 **RAG-PLM**（主线，推理可开/关）

* **检索**：用 **MMseqs2** 为每条序列取 top-K 同源/微型 MSA（K≈32/64），**成本远低于 PSI-BLAST**；已有结果显示 profile 搜索可达 PSI-BLAST 灵敏度的**百倍-400倍加速**；GPU 版 ColabFold 流水线更快。([GitHub][6])
* **融合**（三选一，做消融）：

  1. **Cross-attention** 条带（类似 RAG-ESM）；
  2. **侧通道注意力**对齐同源对位信息；
  3. **Late-fusion** 统计（共进化打分、简化 PSSM 片段）。
* **依据**：AIDO.RAGPLM / RAGFold 与 **RAG-ESM** 的报告。([BioRxiv][2])

### 第三步：家族条件 **Adapter/LoRA**（替代大 MoE）

* **条件信号**：InterPro/Pfam 家族 one-hot 或概率分布（InterProScan 6 产出；注意标注版本以便复现）。([interproscan-docsdev.readthedocs.io][7])
* **放置位置**：输入投影与最后一层前（两处）；
* **目的**：在不引入 MoE 负载不均和长尾碎片化的前提下，引入“专才化”。

### 第四步：**图小波**（尺度模块，放在图上）

* **做法**：在 PPI 图上施加 **Graph Wavelet/SGWT（Chebyshev 近似）**，t∈{0.5,1,2,4}；输出多尺度特征堆叠回节点。([arXiv][3])
* **动机**：显式编码“邻域半径/分辨率”差异，强化“尺度—家族”的交互；近期 PPI 研究已将**超曲率 GNN × 图小波**组合为统一架构并报告收益（HyboWaveNet/HyWinNet）。([arXiv][8])

### 第五步：**洛伦兹模型的超曲率 GNN**（可学习曲率）

* **选择**：H2H-GCN/L²GCN/LGIN 一类**全双曲/洛伦兹**操作，避免频繁回到切空间；**曲率 κ 可学习**（按层或按层级）。([CVF开放获取][4])
* **补充**：HCGNN/ACE-HGNN 展示了基于离散/连续曲率的**自适应曲率**思路，可纳入对照。([arXiv][9])

### （可选）第六步：**PSSM 作为特权信息/解释支路（LUPI/KD）**

* **使用方式**：

  * **训练期**：以高质 PSSM 为 teacher，对学生（仅 PLM+RAG）做**蒸馏**；
  * **评测期**：仅在小样本/偏分布子集上拼接一个**轻量 PSSM 统计向量**，用于量化其稳健/解释增益；
* **推理**：默认**不依赖** PSSM。这样既享受 RAG 的增益，也保留“可解释/稳健”证据链。
* **背景**：已有工作用 **ESM-2 替代 PSSM** 且性能不降，为“PSSM 降级”提供正当性。([BioRxiv][10])

---

## 4. 预期优势

1. **性能与可扩展性**：PLM 主通道 + RAG 注入共进化，较纯单序列更强，同时避免全量 MSA 昂贵代价。([BioRxiv][2])
2. **几何契合**：图小波显式“尺度”，洛伦兹 GNN 承载“层级”，二者互补；已在 PPI 任务中出现联合范式先例。([arXiv][8])
3. **家族专才化且工程友好**：条件 Adapter/LoRA 取代大 MoE，参数效率与稳定性更佳（InterPro/Pfam 标注成熟且更新频繁）。([interproscan-docsdev.readthedocs.io][11])
4. **可解释性链条**：同源检索权重 → 图小波尺度能量 → 洛伦兹半径/角坐标分布；再辅以 PSSM-LUPI 的位点/尺度证据。

---

## 5. 实施挑战

* **数据切分与负样本**：严格采用 **protein-/family-disjoint** 或接口/结构聚类切分；负样本采用**拓扑驱动/聚类**策略并报告敏感性，规避“虚高”。([arXiv][12])
* **检索质量与 K 值选择**：RAG 增益受同源质量影响，需做 **K-sensitivity** 与“无同源”子集评估。([OpenReview][13])
* **几何数值稳定性**：全双曲/洛伦兹操作在优化上更稳，但仍需注意指数/对数映射的数值域与梯度裁剪；可参考 H2H-GCN/L²GCN/SRBGCN 实践。([CVF开放获取][4])
* **注释版本复现**：InterPro/Pfam 版本更新快，**固定 InterProScan/InterPro 版本号**并在日志中输出。([interproscan-docsdev.readthedocs.io][7])
* **算力/工程**：优先 **MMseqs2（GPU）** 流水线以控制检索成本。([NVIDIA Developer][14])

---

## 6. 结论

该方案在**主干**上坚决“去 MSA 压力”（PLM + RAG），在**几何/尺度**上“强耦合”（图小波 × 洛伦兹/可学曲率），并用**家族条件 Adapter/LoRA**实现轻量专才化；**PSSM**被降级为**特权信息/解释支路**以支撑稳健性证据。整体既面向 **SOTA 性能**，又保持**工程可落地**与**审稿可解释**的叙事闭环。

[1]: https://www.biorxiv.org/content/10.1101/2024.07.01.600583v1?utm_source=chatgpt.com "Simulating 500 million years of evolution with a language model"
[2]: https://www.biorxiv.org/content/10.1101/2024.12.02.626519v1?utm_source=chatgpt.com "Retrieval Augmented Protein Language Models for Protein Structure ..."
[3]: https://arxiv.org/abs/1904.07785?utm_source=chatgpt.com "Graph Wavelet Neural Network"
[4]: https://openaccess.thecvf.com/content/CVPR2021/papers/Dai_A_Hyperbolic-to-Hyperbolic_Graph_Convolutional_Network_CVPR_2021_paper.pdf?utm_source=chatgpt.com "A Hyperbolic-to-Hyperbolic Graph Convolutional Network"
[5]: https://academic.oup.com/bioinformaticsadvances/article/4/1/vbad191/7511844?utm_source=chatgpt.com "DeepRank-GNN-esm: a graph neural network for scoring protein–protein ..."
[6]: https://github.com/soedinglab/MMseqs2?utm_source=chatgpt.com "GitHub - soedinglab/MMseqs2: MMseqs2: ultra fast and sensitive search ..."
[7]: https://interproscan-docsdev.readthedocs.io/en/stable/MigratingToI6.html?utm_source=chatgpt.com "Migrating from InterProScan Version 5 to Version 6"
[8]: https://arxiv.org/pdf/2504.20102?utm_source=chatgpt.com "arXiv:2504.20102v1 [cs.LG] 27 Apr 2025"
[9]: https://arxiv.org/pdf/2212.01793v1?utm_source=chatgpt.com "Hyperbolic Curvature Graph Neural Network"
[10]: https://www.biorxiv.org/content/10.1101/2023.06.22.546080v1?utm_source=chatgpt.com "DeepRank-GNN-esm: A Graph Neural Network for Scoring Protein-Protein ..."
[11]: https://interproscan-docsdev.readthedocs.io/en/latest/ReleaseNotes.html?utm_source=chatgpt.com "Release notes: InterProScan 5.75-106.0"
[12]: https://arxiv.org/abs/2404.10457?utm_source=chatgpt.com "Revealing data leakage in protein interaction benchmarks"
[13]: https://openreview.net/forum?id=i4vevaqugi&utm_source=chatgpt.com "RAG-ESM: Improving pretrained protein language models via sequence ..."
[14]: https://developer.nvidia.com/zh-cn/blog/boost-alphafold2-protein-structure-prediction-with-gpu-accelerated-mmseqs2/?utm_source=chatgpt.com "借助 GPU 加速的 MMseqs2 提升 AlphaFold2 蛋白质结构 ..."
[15]: https://academic.oup.com/bib/article/25/2/bbae076/7621029?utm_source=chatgpt.com "Cracking the black box of deep sequence-based protein–protein ..."
[16]: https://openreview.net/forum?id=vTjC0BwKRI&utm_source=chatgpt.com "L\²GC: Lorentzian Linear Graph Convolutional Networks for Node"