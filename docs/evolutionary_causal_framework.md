# 进化动力学与因果干预混合框架：用于蛋白质相互作用预测

**版本**: 1.0  
**日期**: 2025年10月  
**作者**: Hyperbolic_Big项目组  
**目标期刊**: Nature / Nature Methods

---

## 📋 目录

1. [核心创新与动机](#1-核心创新与动机)
2. [理论基础](#2-理论基础)
3. [方法框架](#3-方法框架)
4. [技术实现](#4-技术实现)
5. [数据需求](#5-数据需求)
6. [实验设计](#6-实验设计)
7. [预期结果与影响](#7-预期结果与影响)
8. [实施路线图](#8-实施路线图)
9. [潜在挑战与解决方案](#9-潜在挑战与解决方案)

---

## 1. 核心创新与动机

### 1.1 现有方法的根本缺陷

**问题1: 静态视角的局限**
```
现有方法: Protein A + Protein B → PPI (yes/no, static)
生物现实: PPI是数亿年进化的产物，是动态、受约束的过程
```

- 忽略了PPI的**进化历史**
- 无法解释**为什么**某些PPI存在
- 无法预测**未来**或**新环境**下的PPI

**问题2: 相关性vs因果性**
```
训练数据: 观察性PPI网络 (correlational)
实际需求: 预测干预效应 (causal)
         例如：突变、药物、环境变化
```

- 现有模型学习的是**表面关联**
- 在分布外数据（OOD）上泛化能力差
- 无法回答干预性问题（"what if"）

**问题3: 缺乏可解释的生物学机制**
```
黑盒预测 → 无法理解生物学原理
         → 无法指导实验设计
         → 无法应用于蛋白质工程
```

### 1.2 核心创新：从"预测存在"到"理解机制"

> **Central Hypothesis (核心假设):**  
> PPI不是独立的二元标签，而是进化动力系统在因果约束下的涌现属性。通过联合建模进化轨迹与因果机制，我们可以预测、解释并操纵PPI。

#### 三个关键转变

| 传统范式 | 新范式 | 影响 |
|---------|--------|------|
| 静态预测 | 动态建模 | 理解PPI的进化历史和未来 |
| 学习相关性 | 推断因果性 | 预测干预效应（突变、药物）|
| 黑盒模型 | 机制模型 | 可解释、可指导实验 |

### 1.3 为什么是Nature级别？

**理论贡献**
- 首次将**进化生物学、因果推断、深度学习**统一到单一框架
- 提出PPI的**动力系统理论**
- 解决长期悬而未决的问题：如何从观察性数据推断因果机制

**方法创新**
- 不是"拼装现有模块"，而是**全新范式**
- 利用进化 = 自然的长期干预实验
- 首个能预测**反事实PPI**的模型

**生物学影响**
- 回答根本问题：PPI的**进化起源**和**维持机制**
- 发现新的生物学规律（进化约束模式）
- 实际应用：蛋白质设计、疾病机制、药物靶点

**技术影响**
- 可推广到其他生物网络（基因调控、代谢）
- 为"如何从观察性数据学习因果模型"提供范例
- 开创"进化深度学习"新领域

---

## 2. 理论基础

### 2.1 PPI的进化动力学模型

#### 2.1.1 数学形式化

将PPI视为连续时间马尔可夫过程：

$$
\frac{dP_{AB}(t)}{dt} = f(S_A(t), S_B(t), \theta_{\text{evo}}, \theta_{\text{sel}})
$$

其中：
- $P_{AB}(t)$: 时间$t$时蛋白A和B的相互作用概率/强度
- $S_A(t), S_B(t)$: A和B的序列在时间$t$的状态
- $\theta_{\text{evo}}$: 进化过程参数（突变率、重组率等）
- $\theta_{\text{sel}}$: 选择压力参数（从数据学习）

#### 2.1.2 生物学解释

**进化轨迹的三个阶段**：
```
1. 祖先状态 (t=0): 共同祖先的序列和PPI状态
2. 进化过程 (0<t<T): 突变积累 + 选择筛选
3. 现代状态 (t=T): 观察到的现代物种PPI
```

**关键洞察**：
- PPI的存在不是随机的，而是**进化选择**的结果
- 保守的PPI → 强功能约束 → 高负选择
- divergent PPI → 弱约束或正选择

#### 2.1.3 共进化耦合

两个蛋白质的进化不是独立的：

$$
P(S_A(t), S_B(t) | \text{PPI exists}) \neq P(S_A(t)) \cdot P(S_B(t))
$$

**协同进化模式**：
- **补偿突变** (Compensatory mutations): A的突变被B的突变补偿
- **共适应** (Co-adaptation): 界面残基协同优化
- **功能耦合** (Functional coupling): 保持相互作用强度

**数学建模**：Potts模型
$$
E(S_A, S_B) = \sum_{i \in A, j \in B} J_{ij}(s_i, s_j) + \sum_{i} h_i(s_i)
$$

### 2.2 因果推断框架

#### 2.2.1 PPI的因果图 (Causal DAG)

```
序列A → 结构A ↘
                界面 → 亲和力 ↘
序列B → 结构B ↗              PPI
                              ↗
稳定性A ──────────────────────┘
稳定性B ──────────────────────┘
  ↑
进化历史（混杂因素）
```

**因果关系**：
- 序列 → 结构 → 界面 → 亲和力 → PPI
- 进化历史是**混杂因素** (confounder)

#### 2.2.2 干预 (Intervention)

**形式化定义**：
- 观察性分布: $P(PPI | A, B)$
- 干预后分布: $P(PPI | do(A'), B)$ （强制改变A到A'）

**干预类型**：
1. **序列干预**: 点突变、插入、缺失
2. **表达干预**: CRISPR敲除/敲低
3. **环境干预**: pH、温度、药物
4. **进化干预**: 物种分化、选择压力改变

#### 2.2.3 反事实推理 (Counterfactual Reasoning)

**反事实问题**：
- "如果蛋白A的残基123突变为丙氨酸，PPI会如何变化？"
- "如果没有进化压力X，这个PPI还会存在吗？"

**三步推理** (Pearl's Ladder of Causation):
```
1. Association (关联): P(PPI|A,B) 
   → 现有方法能做

2. Intervention (干预): P(PPI|do(A'))
   → 本方法的核心

3. Counterfactual (反事实): P(PPI_A' | A, B, PPI_observed)
   → 最高层次的因果推理
```

### 2.3 进化作为自然的干预实验

**关键洞察**：
> 每次物种分化 = 一次自然的"干预实验"  
> 数百万年的进化 = 大规模的"A/B测试"

#### 2.3.1 利用多物种数据进行因果发现

**自然实验的优势**：
- 时间尺度长（真实的长期效应）
- 重复次数多（多个独立进化谱系）
- "随机化"（遗传漂变）

**挑战**：
- 观察不到祖先状态（需要重建）
- 存在混杂因素（背景突变）
- 选择偏差（只能观察到存活的谱系）

#### 2.3.2 从进化轨迹推断因果效应

**核心思想**：如果突变X总是伴随PPI的变化，且这个模式在多个独立谱系重复出现，那么X很可能是PPI的**因果决定因素**。

**数学形式**：
$$
\text{Causal Effect} = \mathbb{E}[PPI(t+1) | \text{Mutation X at t}] - \mathbb{E}[PPI(t+1) | \text{No Mutation}]
$$

在多个进化谱系上平均，控制混杂因素。

---

## 3. 方法框架

### 3.1 整体架构

```
输入: 
  - 多物种序列数据 + 物种树
  - 现代PPI网络
  - (可选) 干预实验数据

     ↓
┌─────────────────────────────────────────┐
│  模块1: 进化轨迹重建                     │
│  - 祖先序列重建                          │
│  - PPI状态演化建模                       │
│  - 共进化耦合学习                        │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  模块2: 因果图学习                       │
│  - 从进化数据发现因果关系                 │
│  - 学习因果图结构                        │
│  - 估计因果效应大小                      │
└─────────────────────────────────────────┘
     ↓
┌─────────────────────────────────────────┐
│  模块3: 干预预测                         │
│  - 序列干预 → PPI变化                    │
│  - 反事实推理                            │
│  - 不确定性量化                          │
└─────────────────────────────────────────┘
     ↓
输出:
  - PPI预测 (现代 + 跨物种)
  - 干预效应预测 (突变、药物)
  - 进化约束图谱
  - 因果解释
```

### 3.2 模块1: 进化轨迹重建

#### 3.2.1 祖先序列重建 (Ancestral Sequence Reconstruction)

**输入**：
- 现代物种序列: $\{S_A^{(1)}, S_A^{(2)}, \ldots, S_A^{(N)}\}$
- 物种树: $\mathcal{T}$ (拓扑 + 分支长度)

**输出**：
- 祖先序列: $\{S_A^{\text{anc}_1}, S_A^{\text{anc}_2}, \ldots\}$ (每个内部节点)

**方法**：联合最大似然 + 深度学习

**传统方法** (PAML, IQ-TREE):
$$
S^{\text{anc}} = \arg\max_{S} P(S | \text{descendants}, \mathcal{T}, \theta_{\text{evo}})
$$

**问题**：
- 假设简单的进化模型（如GTR）
- 不考虑PPI约束
- 忽略共进化

**我们的改进**：Deep Ancestral Reconstruction
```python
class DeepAncestralReconstructor(nn.Module):
    def __init__(self):
        self.sequence_encoder = ESM2()  # 预训练PLM
        self.tree_encoder = TreeLSTM()  # 编码物种树
        self.evolution_model = NeuralODE()  # 连续时间进化
        self.ppi_constraint = PPIAwareDecoder()  # 考虑PPI约束
    
    def forward(self, modern_sequences, tree, ppi_network):
        # 1. 编码现代序列
        seq_embeddings = self.sequence_encoder(modern_sequences)
        
        # 2. 编码物种树结构
        tree_context = self.tree_encoder(tree)
        
        # 3. 后向传播：从叶节点推断祖先
        ancestral_embeddings = self.backward_pass(
            seq_embeddings, tree_context
        )
        
        # 4. 考虑PPI约束：如果A-B在现代有PPI，
        #    祖先也应该有兼容的界面
        ancestral_embeddings = self.ppi_constraint(
            ancestral_embeddings, ppi_network
        )
        
        # 5. 解码为祖先序列（采样多个可能的祖先）
        ancestral_sequences = self.decoder(ancestral_embeddings)
        
        return ancestral_sequences  # 分布，不是单点估计
```

**创新点**：
- ✅ 利用PLM的进化知识
- ✅ 同时考虑序列和PPI约束
- ✅ 输出不确定性（祖先序列分布）
- ✅ 端到端可微分（与下游任务联合训练）

#### 3.2.2 PPI状态演化建模

**目标**：沿进化树追踪PPI状态的变化

**模型**：连续时间马尔可夫链 + 神经网络

$$
P(\text{PPI}_{AB}(t+dt) = 1 | \text{state}(t)) = \lambda_{\text{gain}}(t) \cdot dt
$$
$$
P(\text{PPI}_{AB}(t+dt) = 0 | \text{state}(t)) = \lambda_{\text{loss}}(t) \cdot dt
$$

其中转移率由神经网络参数化：

```python
class PPIEvolutionModel(nn.Module):
    def forward(self, seq_A_t, seq_B_t, ppi_t, branch_length):
        # 计算PPI获得/丢失的速率
        lambda_gain = self.gain_network(seq_A_t, seq_B_t, context)
        lambda_loss = self.loss_network(seq_A_t, seq_B_t, context)
        
        # 沿分支演化PPI状态
        ppi_t_plus_dt = self.evolve(ppi_t, lambda_gain, lambda_loss, 
                                     branch_length)
        
        return ppi_t_plus_dt, lambda_gain, lambda_loss
```

**学习目标**：
$$
\mathcal{L}_{\text{evo}} = \sum_{\text{branches}} -\log P(\text{PPI}_{\text{child}} | \text{PPI}_{\text{parent}}, \text{branch})
$$

#### 3.2.3 共进化耦合学习

**问题**：两个蛋白质的进化轨迹相互耦合

**解决方案**：联合建模

```python
class CoevolutionModel(nn.Module):
    def __init__(self):
        # Potts模型参数（残基-残基耦合）
        self.coupling_matrix = nn.Parameter(torch.randn(L_A, L_B, 20, 20))
        self.fields = nn.Parameter(torch.randn(L_A + L_B, 20))
    
    def energy(self, seq_A, seq_B):
        # 计算序列对的能量（越低越稳定）
        E_coupling = sum(self.coupling_matrix[i,j, seq_A[i], seq_B[j]] 
                        for i in range(L_A) for j in range(L_B))
        E_fields = sum(self.fields[i, seq_A[i]] for i in range(L_A)) + \
                   sum(self.fields[j+L_A, seq_B[j]] for j in range(L_B))
        return E_coupling + E_fields
    
    def coevolution_pressure(self, seq_A, seq_B):
        # 计算共进化压力（梯度）
        energy = self.energy(seq_A, seq_B)
        grad_A = torch.autograd.grad(energy, seq_A)[0]
        grad_B = torch.autograd.grad(energy, seq_B)[0]
        return grad_A, grad_B
```

**整合到进化模型**：
- 共进化能量 → 影响突变接受率
- 补偿突变更容易被固定
- 破坏界面的突变被选择清除

### 3.3 模块2: 因果图学习

#### 3.3.1 从进化数据发现因果关系

**核心思想**：时间顺序 + 重复观察 → 因果推断

**算法**：Evolutionary Causal Discovery

```python
class EvolutionaryCausalDiscovery:
    def discover_causal_graph(self, evolutionary_trajectories):
        """
        从多个进化谱系发现因果关系
        
        输入: 
          - evolutionary_trajectories: List of (mutations, ppi_changes)
            每个谱系是一个时间序列
        
        输出:
          - causal_graph: DAG表示因果关系
          - effect_sizes: 因果效应大小
        """
        
        # 1. 对齐多个谱系（处理不同的进化速率）
        aligned_trajectories = self.align_trajectories(
            evolutionary_trajectories
        )
        
        # 2. 识别重复出现的模式
        # "突变X → PPI变化"在多个独立谱系出现
        recurring_patterns = self.find_recurring_patterns(
            aligned_trajectories
        )
        
        # 3. 控制混杂因素（背景突变）
        causal_effects = {}
        for pattern in recurring_patterns:
            effect = self.estimate_causal_effect(
                pattern, 
                adjust_for_confounders=True  # 倾向评分匹配
            )
            if effect.is_significant():
                causal_effects[pattern.mutation] = effect
        
        # 4. 构建因果图
        causal_graph = self.build_dag(causal_effects)
        
        return causal_graph, causal_effects
```

**统计方法**：
- **倾向评分匹配** (Propensity Score Matching): 控制背景突变
- **工具变量** (Instrumental Variables): 利用遗传漂变作为"随机化"
- **断点回归** (Regression Discontinuity): 利用选择压力的突变阈值

#### 3.3.2 结构因果模型 (Structural Causal Model)

**定义PPI的SCM**：

$$
\begin{align}
\text{Structure}_A &= f_1(\text{Sequence}_A, U_1) \\
\text{Structure}_B &= f_2(\text{Sequence}_B, U_2) \\
\text{Interface} &= f_3(\text{Structure}_A, \text{Structure}_B, U_3) \\
\text{PPI} &= f_4(\text{Interface}, \text{Stability}_A, \text{Stability}_B, U_4)
\end{align}
$$

其中$U_i$是未观察的外生变量（噪声）

**用神经网络参数化**：

```python
class StructuralCausalModel(nn.Module):
    def __init__(self):
        self.f1 = SequenceToStructure()  # 可以用AlphaFold
        self.f2 = SequenceToStructure()
        self.f3 = InterfacePredictor()
        self.f4 = PPIPredictor()
    
    def forward(self, seq_A, seq_B, noise=None):
        # 正向因果链
        struct_A = self.f1(seq_A, noise.U1 if noise else None)
        struct_B = self.f2(seq_B, noise.U2 if noise else None)
        interface = self.f3(struct_A, struct_B, noise.U3 if noise else None)
        ppi = self.f4(interface, struct_A, struct_B, noise.U4 if noise else None)
        return ppi
    
    def intervene(self, seq_A, seq_B, intervention):
        """
        干预：强制改变某个变量
        """
        if intervention.target == "seq_A":
            seq_A = intervention.value  # do(Sequence_A = new_value)
        
        # 重新计算下游变量（不改变噪声）
        return self.forward(seq_A, seq_B, noise=intervention.noise)
    
    def counterfactual(self, seq_A, seq_B, ppi_observed, intervention):
        """
        反事实推理：
        1. 后向推断噪声（从观察数据）
        2. 前向预测反事实世界
        """
        # 步骤1: 反演噪声
        noise = self.abduction(seq_A, seq_B, ppi_observed)
        
        # 步骤2: 干预
        seq_A_cf = intervention.value if intervention.target == "seq_A" else seq_A
        seq_B_cf = intervention.value if intervention.target == "seq_B" else seq_B
        
        # 步骤3: 预测反事实PPI（使用相同的噪声）
        ppi_cf = self.forward(seq_A_cf, seq_B_cf, noise=noise)
        
        return ppi_cf
```

#### 3.3.3 因果效应估计

**平均因果效应** (Average Causal Effect, ACE):
$$
\text{ACE} = \mathbb{E}[\text{PPI}(do(\text{Mutation}=1))] - \mathbb{E}[\text{PPI}(do(\text{Mutation}=0))]
$$

**条件因果效应** (Conditional ACE):
$$
\text{CACE}(c) = \mathbb{E}[\text{PPI}(do(M=1)) - \text{PPI}(do(M=0)) | \text{Context}=c]
$$

例如：突变对PPI的影响**依赖于**进化背景、家族、物种

**实现**：

```python
def estimate_causal_effect(self, mutation, data):
    # 使用双重机器学习 (Double ML) 消除混杂偏差
    
    # 步骤1: 预测处理变量（突变）
    propensity = self.predict_propensity(mutation, data.confounders)
    
    # 步骤2: 预测结果变量（PPI）
    outcome_pred = self.predict_outcome(data.confounders)
    
    # 步骤3: 计算残差
    treatment_residual = mutation - propensity
    outcome_residual = data.ppi - outcome_pred
    
    # 步骤4: 回归残差（得到无偏估计）
    causal_effect = np.cov(treatment_residual, outcome_residual)[0,1] / \
                    np.var(treatment_residual)
    
    return causal_effect
```

### 3.4 模块3: 干预预测

#### 3.4.1 序列干预预测

**任务**：给定突变，预测PPI变化

```python
class InterventionPredictor(nn.Module):
    def __init__(self, scm):
        self.scm = scm  # 结构因果模型
        self.evolution_model = PPIEvolutionModel()  # 进化约束
    
    def predict_mutation_effect(self, seq_A, seq_B, ppi_obs, mutation):
        """
        预测点突变的效应
        
        输入:
          - seq_A, seq_B: 野生型序列
          - ppi_obs: 观察到的野生型PPI
          - mutation: (protein, position, new_aa)
        
        输出:
          - delta_ppi: PPI变化
          - confidence: 置信区间
          - mechanism: 因果路径（可解释性）
        """
        
        # 1. 反事实推理：如果突变发生
        seq_A_mut = self.apply_mutation(seq_A, mutation)
        ppi_cf = self.scm.counterfactual(
            seq_A, seq_B, ppi_obs, 
            intervention=Intervention("seq_A", seq_A_mut)
        )
        
        # 2. 考虑进化约束：这个突变在进化上可行吗？
        evo_plausibility = self.evolution_model.mutation_fitness(
            seq_A, mutation
        )
        
        # 3. 多路径因果效应分解
        effects = self.decompose_causal_paths(seq_A, seq_A_mut, ppi_obs)
        # effects = {
        #   'structure_change': 0.3,  # 结构改变的贡献
        #   'stability_change': -0.1, # 稳定性的贡献
        #   'interface_change': 0.5,  # 界面的贡献
        # }
        
        # 4. 不确定性量化（贝叶斯）
        ppi_cf_samples = self.sample_counterfactual(
            seq_A, seq_B, ppi_obs, mutation, n_samples=1000
        )
        confidence_interval = np.percentile(ppi_cf_samples, [2.5, 97.5])
        
        return {
            'delta_ppi': ppi_cf - ppi_obs,
            'confidence_interval': confidence_interval,
            'causal_mechanism': effects,
            'evolutionary_plausibility': evo_plausibility
        }
```

#### 3.4.2 药物干预预测

**扩展到小分子干预**：

```python
class DrugInterventionPredictor:
    def predict_drug_effect(self, protein_A, protein_B, drug):
        """
        预测小分子药物对PPI的影响
        
        假设：药物通过改变蛋白质的构象/稳定性来影响PPI
        """
        
        # 1. 预测药物结合位点和亲和力
        binding_site, affinity = self.predict_drug_binding(protein_A, drug)
        
        # 2. 预测药物结合后的构象变化
        conformational_change = self.predict_conformational_change(
            protein_A, drug, binding_site
        )
        
        # 3. 因果链：药物 → 构象 → 界面 → PPI
        ppi_change = self.scm.intervene(
            protein_A, protein_B,
            intervention=Intervention("conformation_A", conformational_change)
        )
        
        return ppi_change
```

#### 3.4.3 进化外推预测

**任务**：预测新物种/新环境下的PPI

```python
def predict_ppi_in_new_species(self, protein_A, protein_B, target_species):
    """
    预测PPI在新物种中是否存在
    
    思路：
    1. 找到与目标物种最近的已知物种
    2. 利用进化模型"外推"到新物种
    3. 考虑物种特异的选择压力
    """
    
    # 1. 在物种树上定位
    closest_species = self.find_closest_species(target_species)
    evolutionary_distance = self.compute_distance(closest_species, target_species)
    
    # 2. 模拟进化过程
    seq_A_target = self.evolve_sequence(
        protein_A, evolutionary_distance, target_species.selection_pressure
    )
    seq_B_target = self.evolve_sequence(
        protein_B, evolutionary_distance, target_species.selection_pressure
    )
    
    # 3. 预测目标物种的PPI
    ppi_target = self.scm.forward(seq_A_target, seq_B_target)
    
    # 4. 评估预测的不确定性（进化随机性）
    uncertainty = self.evolutionary_uncertainty(evolutionary_distance)
    
    return ppi_target, uncertainty
```

---

## 4. 技术实现

### 4.1 深度学习架构

#### 4.1.1 整体架构图

```
输入层:
├── 序列数据 (多物种)
├── 物种树
└── PPI网络

编码层:
├── ESM-2/ESM-3 (序列编码)
├── TreeLSTM (物种树编码)
└── GNN (PPI网络编码)

进化层:
├── 祖先重建网络
├── 神经ODE (连续时间进化)
└── 共进化Potts模型

因果层:
├── 因果图学习
├── 结构因果模型 (SCM)
└── 反事实推理模块

预测层:
├── PPI分类器
├── 干预效应预测器
└── 不确定性量化
```

#### 4.1.2 核心模块实现

**神经ODE用于进化建模**：

```python
class NeuralEvolutionODE(nn.Module):
    """
    用神经ODE建模连续时间的序列进化
    """
    def __init__(self, hidden_dim):
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    
    def forward(self, seq_embedding, time_span):
        """
        seq_embedding: 序列的向量表示
        time_span: [t_start, t_end] 进化时间
        """
        from torchdiffeq import odeint
        
        # 定义ODE: dh/dt = f(h, t)
        def ode_dynamics(t, h):
            return self.ode_func(h)
        
        # 求解ODE
        trajectory = odeint(
            ode_dynamics, 
            seq_embedding, 
            time_span,
            method='dopri5'  # Runge-Kutta方法
        )
        
        return trajectory[-1]  # 返回终点状态
```

**变分自编码器用于不确定性**：

```python
class VariationalAncestralReconstructor(nn.Module):
    """
    用VAE建模祖先序列的不确定性
    """
    def __init__(self):
        self.encoder = ESM2Encoder()
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = SequenceDecoder()
    
    def encode(self, modern_sequences, tree):
        # 编码现代序列
        h = self.encoder(modern_sequences)
        h_tree = self.tree_context(tree)
        h_combined = h + h_tree
        
        # 输出潜变量的分布
        mu = self.mu_head(h_combined)
        logvar = self.logvar_head(h_combined)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, modern_sequences, tree):
        mu, logvar = self.encode(modern_sequences, tree)
        z = self.reparameterize(mu, logvar)
        ancestral_seq = self.decode(z)
        return ancestral_seq, mu, logvar
    
    def sample_ancestors(self, modern_sequences, tree, n_samples=100):
        """采样多个可能的祖先序列"""
        mu, logvar = self.encode(modern_sequences, tree)
        samples = []
        for _ in range(n_samples):
            z = self.reparameterize(mu, logvar)
            ancestral_seq = self.decode(z)
            samples.append(ancestral_seq)
        return samples
```

### 4.2 训练策略

#### 4.2.1 多任务学习

联合优化多个目标：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPI}} + \lambda_1 \mathcal{L}_{\text{evo}} + \lambda_2 \mathcal{L}_{\text{causal}} + \lambda_3 \mathcal{L}_{\text{interv}}
$$

其中：
- $\mathcal{L}_{\text{PPI}}$: 现代PPI预测损失（交叉熵）
- $\mathcal{L}_{\text{evo}}$: 进化轨迹重建损失
- $\mathcal{L}_{\text{causal}}$: 因果图学习损失
- $\mathcal{L}_{\text{interv}}$: 干预预测损失（如果有实验数据）

**具体实现**：

```python
def compute_loss(self, batch):
    # 1. PPI预测损失（标准二分类）
    ppi_pred = self.model(batch.seq_A, batch.seq_B)
    loss_ppi = F.binary_cross_entropy(ppi_pred, batch.ppi_label)
    
    # 2. 进化一致性损失
    # 祖先和后代的PPI应该满足进化模型
    ancestral_ppi = self.model.reconstruct_ancestral_ppi(batch.tree)
    loss_evo = self.evolution_consistency_loss(
        ancestral_ppi, batch.modern_ppi, batch.tree
    )
    
    # 3. 因果一致性损失
    # 学到的因果图应该能解释干预实验
    if batch.has_intervention_data:
        pred_effect = self.model.predict_intervention(batch.intervention)
        true_effect = batch.intervention_effect
        loss_causal = F.mse_loss(pred_effect, true_effect)
    else:
        loss_causal = 0
    
    # 4. 共进化损失
    # 相互作用的蛋白质应该有协同进化信号
    loss_coevo = self.coevolution_loss(batch.seq_A, batch.seq_B, batch.ppi_label)
    
    # 总损失
    loss = loss_ppi + 0.1 * loss_evo + 0.2 * loss_causal + 0.05 * loss_coevo
    
    return loss
```

#### 4.2.2 课程学习 (Curriculum Learning)

**逐步增加任务难度**：

```python
class CurriculumTrainer:
    def __init__(self):
        self.stage = 1
    
    def train_epoch(self, epoch):
        if epoch < 10:
            # 阶段1: 只学习现代PPI预测（简单）
            self.train_ppi_only()
        elif epoch < 20:
            # 阶段2: 加入进化约束（中等）
            self.train_ppi_with_evolution()
        elif epoch < 30:
            # 阶段3: 加入因果学习（困难）
            self.train_full_model()
        else:
            # 阶段4: 微调干预预测（最难）
            self.train_with_interventions()
```

#### 4.2.3 对比学习

**学习进化上相似的蛋白质对**：

```python
class ContrastiveEvolutionLearning:
    def __init__(self):
        self.temperature = 0.07
    
    def contrastive_loss(self, protein_A, positives, negatives):
        """
        protein_A: 查询蛋白质
        positives: 进化上相关的蛋白质（同源）
        negatives: 进化上无关的蛋白质
        """
        # 编码
        z_A = self.encoder(protein_A)
        z_pos = [self.encoder(p) for p in positives]
        z_neg = [self.encoder(n) for n in negatives]
        
        # 对比损失（InfoNCE）
        logits_pos = [torch.dot(z_A, z_p) / self.temperature for z_p in z_pos]
        logits_neg = [torch.dot(z_A, z_n) / self.temperature for z_n in z_neg]
        
        logits = torch.cat([torch.stack(logits_pos), torch.stack(logits_neg)])
        labels = torch.zeros(len(logits))
        labels[:len(positives)] = 1
        
        loss = F.cross_entropy(logits, labels)
        return loss
```

### 4.3 工程优化

#### 4.3.1 分布式训练

```python
# 使用PyTorch DDP进行多GPU训练
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train_distributed():
    local_rank = setup_distributed()
    model = EvolutionaryCausalPPIModel().to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    
    # 训练循环...
```

#### 4.3.2 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():  # 自动混合精度
        loss = model.compute_loss(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 4.3.3 梯度累积（处理大batch）

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model.compute_loss(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. 数据需求

### 5.1 核心数据集

#### 5.1.1 PRING基准数据 ✅ (已有)

- **Human**: 训练集（~100K PPI对）
- **跨物种测试**: Arath, Yeast, Ecoli
- **优势**: 标准化、高质量、可对比

#### 5.1.2 扩展多物种数据 ✅ (使用STRING v12.0全集)

**数据来源**: STRING v12.0数据库（服务器已有完整数据）

**目标物种**: 至少**10-15个物种**，覆盖主要进化谱系

| 谱系 | 物种 | NCBI Taxon ID | 进化距离 | 预期PPI数 |
|-----|------|---------------|---------|----------|
| **灵长类** | Human (智人) | 9606 | 基准 | ~200,000 |
| | Chimpanzee (黑猩猩) | 9598 | 6 Mya | ~150,000 |
| | Macaque (猕猴) | 9544 | 25 Mya | ~100,000 |
| **哺乳类** | Mouse (小鼠) | 10090 | 90 Mya | ~180,000 |
| | Rat (大鼠) | 10116 | 90 Mya | ~120,000 |
| | Dog (狗) | 9615 | 95 Mya | ~80,000 |
| | Cow (牛) | 9913 | 95 Mya | ~60,000 |
| **脊椎动物** | Zebrafish (斑马鱼) | 7955 | 450 Mya | ~100,000 |
| | Chicken (鸡) | 9031 | 300 Mya | ~70,000 |
| **植物** | A.thaliana (拟南芥) | 3702 | 1,500 Mya | ~150,000 |
| | Rice (水稻) | 4530 | 1,500 Mya | ~80,000 |
| **真菌** | S.cerevisiae (酵母) | 4932 | 1,000 Mya | ~200,000 |
| | S.pombe (裂殖酵母) | 4896 | 1,000 Mya | ~60,000 |
| **细菌** | E.coli (大肠杆菌) | 511145 | 3,000 Mya | ~100,000 |
| | B.subtilis (枯草芽孢杆菌) | 224308 | 3,000 Mya | ~50,000 |

**物种树（简化）**:
```
                         ┌─ Human
                    ┌────┤ Chimp (6 Mya)
               ┌────┤    └─ Macaque (25 Mya)
          ┌────┤    │
          │    │    └──────── Mouse, Rat (90 Mya)
          │    │
    ──────┤    └────────────── Zebrafish (450 Mya)
          │
          │         ┌────────── Arath, Rice (1,500 Mya)
          └─────────┤
                    │    ┌───── Yeast, S.pombe (1,000 Mya)
                    └────┤
                         └───── E.coli, B.subtilis (3,000 Mya)
```

---

### 🎯 STRING数据过滤策略（重要）

#### 策略1: Combined Score阈值选择

**STRING评分系统**：
- **范围**: 0-1000（综合7个证据通道）
- **7个通道**: Neighborhood, Fusion, Co-occurrence, Co-expression, **Experimental**, **Database**, Text mining

**评分质量对照表**：

| Score阈值 | 置信度等级 | 预期假阳性率 | 数据量 (Human) | 推荐用途 |
|----------|-----------|------------|---------------|---------|
| 900-1000 | 最高 | ~10% | ~50,000 | 核心训练集 |
| **700-900** | **高** | **~20%** | **~200,000** | **✅ 标准训练集（推荐）** |
| 400-700 | 中 | ~40% | ~500,000 | 扩展/进化分析 |
| 150-400 | 低 | ~60% | ~1,000,000 | ⚠️ 不推荐 |

**推荐阈值**: **Combined Score >= 700**

**理由**：
1. ✅ **平衡质量与数量**: ~20%假阳性率可接受，数据量充足
2. ✅ **覆盖多种证据**: 包含实验、数据库、共表达等多种证据
3. ✅ **支持统计推断**: 数据量足够训练深度学习模型
4. ✅ **包含进化信号**: 共现、共表达包含重要的进化相关信息

#### 策略2: 是否使用"物理子网"？

**物理子网定义**: 仅包含有直接物理接触证据的PPI（Experimental + Database通道有值）

**对比分析**：

| 指标 | 物理子网 | Combined Score >= 700 |
|-----|---------|---------------------|
| **数据量** (Human) | ~30,000 | ~200,000 |
| **假阳性率** | ~15% | ~20% |
| **质量提升** | +5% | 基准 |
| **进化信号** | ❌ 弱（丢失共表达、共现） | ✅ 强（包含多种证据） |
| **偏向性** | ❌ 严重（研究热点过度代表） | ✅ 较小 |
| **推荐度** | ⚠️ 不推荐 | ✅ 推荐 |

**结论**: ❌ **不使用物理子网，使用完整的Combined Score过滤**

**原因**：
1. 物理子网数据量太少（仅15%），无法支持深度学习
2. 质量提升有限（仅5%），但损失85%数据
3. 丢失了共表达、共现等重要的进化信号
4. 过度代表研究热点蛋白，泛化能力差

#### 策略3: 分层训练策略（推荐）

**三层数据划分**：

```python
class StratifiedStringDataset:
    """分层STRING数据集"""
    
    def __init__(self):
        # 三个置信度层
        self.tiers = {
            'tier1_core': {
                'threshold': 900,
                'weight': 1.0,
                'purpose': '核心训练集（高质量）'
            },
            'tier2_standard': {
                'threshold': 700,
                'weight': 1.0,
                'purpose': '标准训练集（平衡）'
            },
            'tier3_extended': {
                'threshold': 400,
                'weight': 0.5,
                'purpose': '扩展集（进化分析）'
            }
        }
    
    def load_ppi_data(self, species, taxon_id):
        """
        从STRING加载PPI数据
        
        参数:
            species: 物种名称
            taxon_id: NCBI Taxon ID
        
        返回:
            分层的PPI数据
        """
        
        # 从STRING全集中提取该物种的数据
        all_ppis = self.load_from_string_db(taxon_id)
        
        # 按置信度分层
        tier1 = all_ppis[all_ppis['combined_score'] >= 900]
        tier2 = all_ppis[all_ppis['combined_score'] >= 700]
        tier3 = all_ppis[all_ppis['combined_score'] >= 400]
        
        return {
            'tier1': tier1,  # 核心集
            'tier2': tier2,  # 标准集（包含tier1）
            'tier3': tier3   # 扩展集（包含tier1+tier2）
        }
    
    def compute_loss(self, predictions, labels, tier):
        """
        根据数据层调整损失权重
        """
        weight = self.tiers[tier]['weight']
        loss = F.binary_cross_entropy(predictions, labels)
        return weight * loss
```

**训练流程**：
```python
# 训练循环
for epoch in range(num_epochs):
    # 1. 核心集 + 标准集（权重1.0）
    loss1 = train_on_data(tier1_data, weight=1.0)
    loss2 = train_on_data(tier2_data, weight=1.0)
    
    # 2. 扩展集（权重0.5，用于捕捉弱信号）
    loss3 = train_on_data(tier3_data, weight=0.5)
    
    # 3. 总损失
    total_loss = loss1 + loss2 + loss3
```

---

### 📦 数据收集脚本

```python
class MultiSpeciesStringCollector:
    """多物种STRING数据收集器"""
    
    def __init__(self, string_data_path):
        """
        参数:
            string_data_path: STRING数据库路径（服务器上的完整数据）
        """
        self.string_data_path = string_data_path
        self.confidence_threshold = 700  # 标准阈值
        
        # 目标物种列表
        self.target_species = {
            'human': 9606,
            'chimpanzee': 9598,
            'macaque': 9544,
            'mouse': 10090,
            'rat': 10116,
            'dog': 9615,
            'zebrafish': 7955,
            'chicken': 9031,
            'arath': 3702,
            'rice': 4530,
            'yeast': 4932,
            's_pombe': 4896,
            'ecoli': 511145,
            'b_subtilis': 224308
        }
    
    def collect_all_species(self):
        """收集所有物种的PPI数据"""
        
        all_data = {}
        
        for species_name, taxon_id in self.target_species.items():
            print(f"Processing {species_name} (taxon: {taxon_id})...")
            
            # 1. 从STRING提取PPI网络
            ppis = self.extract_ppis(taxon_id)
            
            # 2. 提取序列
            sequences = self.extract_sequences(taxon_id)
            
            # 3. 统计信息
            stats = self.compute_statistics(ppis, sequences)
            
            all_data[species_name] = {
                'taxon_id': taxon_id,
                'ppis': ppis,
                'sequences': sequences,
                'stats': stats
            }
            
            print(f"  ✓ {species_name}: {len(ppis):,} PPIs, "
                  f"{len(sequences):,} proteins")
        
        return all_data
    
    def extract_ppis(self, taxon_id, min_score=700):
        """
        从STRING提取PPI数据
        
        参数:
            taxon_id: NCBI Taxon ID
            min_score: 最小置信度分数
        
        返回:
            DataFrame with columns: protein1, protein2, combined_score, 
                                   neighborhood, fusion, cooccurence, 
                                   coexpression, experimental, database, textmining
        """
        
        # 读取STRING protein.links.detailed.v12.0.txt
        # 格式: protein1 protein2 neighborhood fusion ... combined_score
        
        ppis = []
        
        links_file = f"{self.string_data_path}/{taxon_id}.protein.links.detailed.v12.0.txt"
        
        with open(links_file) as f:
            header = f.readline()  # 跳过表头
            
            for line in f:
                fields = line.strip().split()
                combined_score = int(fields[-1])
                
                # 过滤低置信度
                if combined_score >= min_score:
                    ppis.append({
                        'protein1': fields[0],
                        'protein2': fields[1],
                        'neighborhood': int(fields[2]),
                        'fusion': int(fields[3]),
                        'cooccurence': int(fields[4]),
                        'coexpression': int(fields[5]),
                        'experimental': int(fields[6]),
                        'database': int(fields[7]),
                        'textmining': int(fields[8]),
                        'combined_score': combined_score
                    })
        
        return pd.DataFrame(ppis)
    
    def extract_sequences(self, taxon_id):
        """提取蛋白质序列"""
        
        sequences = {}
        
        seq_file = f"{self.string_data_path}/{taxon_id}.protein.sequences.v12.0.fa"
        
        from Bio import SeqIO
        for record in SeqIO.parse(seq_file, 'fasta'):
            sequences[record.id] = str(record.seq)
        
        return sequences
    
    def compute_statistics(self, ppis, sequences):
        """计算统计信息"""
        
        return {
            'num_ppis': len(ppis),
            'num_proteins': len(sequences),
            'avg_combined_score': ppis['combined_score'].mean(),
            'has_experimental': (ppis['experimental'] > 0).sum(),
            'has_database': (ppis['database'] > 0).sum(),
            'physical_subset': ((ppis['experimental'] > 0) | 
                               (ppis['database'] > 0)).sum()
        }
```

---

### 📊 预期数据规模

| 物种 | Proteins | PPIs (Score>=700) | PPIs (Score>=900) | 物理子网 |
|------|----------|-------------------|-------------------|---------|
| Human | ~20,000 | ~200,000 | ~50,000 | ~30,000 |
| Mouse | ~22,000 | ~180,000 | ~45,000 | ~28,000 |
| Yeast | ~6,000 | ~200,000 | ~80,000 | ~60,000 |
| Arath | ~27,000 | ~150,000 | ~40,000 | ~25,000 |
| Ecoli | ~4,400 | ~100,000 | ~30,000 | ~20,000 |
| **总计** | **~100,000** | **~1,500,000** | **~400,000** | **~250,000** |

**存储需求**: 约50-100GB（压缩后）

#### 5.1.3 干预实验数据 🆕 (需要整合)

**1. 点突变数据**

| 数据库 | 规模 | 内容 | 用途 |
|--------|------|------|------|
| [DeepMutationalScan](https://github.com/OATML-Markslab/DeepSequence) | ~50个蛋白质家族 | 深度突变扫描数据 | 验证突变效应预测 |
| [ProTherm](https://web.iitm.ac.in/bioinfo2/prothermdb/) | ~25,000个突变 | 热稳定性变化 | 稳定性路径因果效应 |
| [SKEMPI 2.0](https://life.bsc.es/pid/skempi2) | ~7,000个突变 | 结合亲和力变化 (ΔΔG) | **直接验证PPI效应** ⭐ |

**SKEMPI示例数据**：
```
Protein1  Protein2  Mutation  ΔΔG(kcal/mol)  Effect
1A22_A    1A22_B    A_T10A    +2.3           Weaken
1BRS_A    1BRS_B    A_Y35F    -0.5           Slightly strengthen
```

**整合脚本**：
```python
class InterventionDataIntegrator:
    def load_skempi(self):
        # 加载SKEMPI数据
        skempi = pd.read_csv("SKEMPI_v2.csv")
        
        # 转换为因果干预格式
        interventions = []
        for _, row in skempi.iterrows():
            interventions.append({
                'protein_A': row['Protein1'],
                'protein_B': row['Protein2'],
                'intervention': {
                    'type': 'mutation',
                    'position': row['Position'],
                    'from_aa': row['Wild_type'],
                    'to_aa': row['Mutant']
                },
                'effect': {
                    'delta_binding_affinity': row['ddG'],
                    'ppi_change': 1 if row['ddG'] < -0.5 else (-1 if row['ddG'] > 0.5 else 0)
                }
            })
        
        return interventions
```

**2. CRISPR筛选数据**

| 数据库 | 规模 | 内容 | 用途 |
|--------|------|------|------|
| [DepMap](https://depmap.org/) | ~1000个细胞系 | 基因敲除对细胞活力的影响 | 网络级干预效应 |
| [CORUM](https://mips.helmholtz-muenchen.de/corum/) | ~5000个复合物 | 蛋白质复合物组成 | 验证多蛋白PPI预测 |

**3. 药物扰动数据**

| 数据库 | 规模 | 内容 | 用途 |
|--------|------|------|------|
| [LINCS L1000](https://lincsproject.org/) | ~100万个基因表达谱 | 小分子化合物扰动 | 间接干预效应 |
| [STITCH](http://stitch.embl.de/) | ~50万个蛋白-化合物相互作用 | 药物靶点 | 药物干预建模 |

### 5.2 辅助数据

#### 5.2.1 结构数据

- **AlphaFold Database**: ~200M 结构预测（用于界面分析）
- **PDB**: ~20万实验结构（高质量界面）
- **PDBbind**: ~20,000个蛋白-配体复合物（结合亲和力）

#### 5.2.2 功能注释

- **GO (Gene Ontology)**: 功能分类
- **InterPro/Pfam**: 蛋白质家族和结构域
- **KEGG Pathway**: 通路信息

#### 5.2.3 进化信息

- **TimeTree**: 物种分化时间
- **OrthoDB**: 直系同源关系
- **Pfam**: 家族进化树

### 5.3 数据预处理流程

```python
class DataPreprocessingPipeline:
    def __init__(self):
        self.species_tree = self.build_species_tree()
    
    def build_species_tree(self):
        """构建物种树（从TimeTree数据库）"""
        from ete3 import Tree
        
        # 下载物种树
        tree = Tree("((human:6,mouse:6):90,zebrafish:96,((arath:150):50,yeast:200):50,ecoli:300);", format=1)
        
        # 标注分支长度（百万年）
        return tree
    
    def align_orthologs(self, species_A, species_B):
        """对齐两个物种的直系同源蛋白"""
        orthologs = []
        
        # 从OrthoDB查询
        orthodb = OrthoDBQuery()
        for protein_A in self.get_proteins(species_A):
            protein_B = orthodb.find_ortholog(protein_A, species_B)
            if protein_B:
                orthologs.append((protein_A, protein_B))
        
        return orthologs
    
    def construct_training_pairs(self):
        """构建训练数据：(seq_A, seq_B, PPI, evolutionary_context)"""
        training_data = []
        
        for species in self.species_list:
            ppi_network = self.load_ppi(species)
            
            for edge in ppi_network.edges():
                protein_A, protein_B = edge
                
                # 查找直系同源
                orthologs_A = self.find_orthologs_across_species(protein_A)
                orthologs_B = self.find_orthologs_across_species(protein_B)
                
                # 构建进化上下文
                evo_context = {
                    'species': species,
                    'orthologs_A': orthologs_A,
                    'orthologs_B': orthologs_B,
                    'conservation_score': self.compute_conservation(orthologs_A, orthologs_B),
                    'species_tree_path': self.species_tree.get_path(species, 'human')
                }
                
                training_data.append({
                    'seq_A': self.get_sequence(protein_A),
                    'seq_B': self.get_sequence(protein_B),
                    'ppi': 1,
                    'evo_context': evo_context
                })
        
        return training_data
```

---

## 6. 实验设计

### 6.1 评估任务

#### 6.1.1 任务1: 标准PPI预测（基线对比）

**目标**: 证明方法在传统任务上不输于现有方法

**设置**:
- 数据集: PRING benchmark (human train/val/test)
- 基线方法: 
  - DeepPPI (PSSM-based)
  - PIPR (sequence-only)
  - GNN-PPI (graph-based)
  - ESM-based baseline
- 评估指标: AUPR, AUROC, F1, Precision@K

**预期结果**: 
- 至少与最佳基线持平
- 在长尾家族上显著提升（因为考虑了进化）

#### 6.1.2 任务2: 跨物种泛化 ⭐ (核心优势)

**目标**: 证明进化建模带来的泛化能力

**设置**:
```
训练: Human PPI网络
测试: Arath, Yeast, Ecoli (零样本)
```

**对比实验**:
- 无进化信息: 直接预测
- 有进化信息: 利用进化轨迹外推

**评估指标**:
- 标准指标: AUPR, AUROC
- **进化一致性**: 预测的PPI是否符合进化约束？
- **系统发育信号**: 预测准确率 vs 进化距离的关系

**预期结果**:
- 传统方法在远缘物种上崩溃
- 我们的方法性能平稳下降（因为显式建模进化）

**关键图表**:
```
性能 (AUPR)
  │
  │    ●────●────●  传统方法（突然崩溃）
  │              ╲
  │               ╲
  │    ●────●────●─●─●  我们的方法（平稳下降）
  │
  └─────────────────────── 进化距离
  0   灵长  哺乳  脊椎  真菌  细菌
```

#### 6.1.3 任务3: 突变效应预测 ⭐⭐ (核心创新)

**目标**: 证明因果建模能预测干预效应

**设置**:
- 测试集: SKEMPI 2.0（7000+个突变，已知ΔΔG）
- 任务: 给定野生型PPI和突变，预测结合亲和力变化

**对比方法**:
- 纯序列方法: ESM-1v (zero-shot mutation effect)
- 结构方法: FoldX, Rosetta
- **我们的方法**: 因果干预预测

**评估指标**:
- Spearman相关系数 (预测ΔΔG vs 实验ΔΔG)
- 分类准确率 (strengthen/weaken/neutral)

**预期结果**:
- 超越纯序列方法（因为考虑了结构和进化）
- 接近或超越基于物理的方法（FoldX）
- **关键优势**: 可以给出因果解释！

**案例研究**:
```
突变: Barnase (A) - Barstar (B), A_R83A

预测:
  ΔΔG = +3.2 kcal/mol (实验值: +3.5)
  
因果解释:
  - 直接效应 (60%): R83盐桥断裂 → 界面亲和力下降
  - 间接效应 (30%): 局部结构松弛 → 进一步削弱界面
  - 进化证据 (10%): R83在所有脊椎动物中保守 → 功能关键
  
置信度: 高（因为类似突变在进化中被清除）
```

#### 6.1.4 任务4: 反事实推理 ⭐⭐⭐ (最高创新)

**目标**: 回答"what if"问题（现有方法无法做到）

**示例问题**:
1. "如果HIV蛋白进化到与人类蛋白有更强PPI，会怎样？"
2. "如果去除某个选择压力，哪些PPI会消失？"
3. "设计一个突变组合，使得PPI强度增加2倍？"

**实验设置**:

**实验4.1: 进化反事实**
```
问题: 如果灵长类没有特定的选择压力（如病毒防御），
     某些PPI是否会消失？

方法:
1. 识别在灵长类中快速进化的PPI
2. 反事实：移除正选择压力
3. 预测PPI的"中性进化"轨迹

验证: 比较与非灵长类哺乳动物的PPI（它们缺少这个压力）
```

**实验4.2: 治疗反事实**
```
问题: 对于致病突变导致的PPI破坏，
     哪些二次突变可以恢复PPI？

方法:
1. 输入：致病突变 + PPI破坏
2. 反事实搜索：找到恢复PPI的突变组合
3. 排序：按进化可行性和效果

应用: 精准医疗、蛋白质工程
```

### 6.2 消融实验

**目标**: 验证每个模块的贡献

| 模型变体 | 进化模块 | 因果模块 | 共进化 | 预期性能 |
|---------|---------|---------|--------|---------|
| Baseline | ✗ | ✗ | ✗ | 基线 |
| +Evolution | ✓ | ✗ | ✗ | +5% AUPR（跨物种） |
| +Causal | ✗ | ✓ | ✗ | +10% 突变预测 |
| +Coevolution | ✗ | ✗ | ✓ | +3% AUPR |
| **Full Model** | ✓ | ✓ | ✓ | **最优** |

**关键消融**:
1. **进化深度**: 比较使用5个 vs 10个 vs 20个物种
2. **因果图结构**: 比较手工设计 vs 自动学习的因果图
3. **不确定性**: 比较点估计 vs 分布估计（VAE）

### 6.3 可解释性分析

#### 6.3.1 进化约束可视化

**绘制PPI的进化轨迹**:
```python
def plot_ppi_evolution_trajectory(protein_A, protein_B):
    # 沿物种树重建PPI强度
    trajectory = model.reconstruct_ppi_trajectory(protein_A, protein_B, species_tree)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory['time'], trajectory['ppi_strength'], 'o-')
    plt.xlabel("Evolutionary Time (Million Years)")
    plt.ylabel("PPI Strength")
    plt.title(f"Evolutionary Trajectory of {protein_A}-{protein_B} Interaction")
    
    # 标注关键事件
    for event in trajectory['key_events']:
        plt.axvline(event['time'], color='red', linestyle='--', alpha=0.5)
        plt.text(event['time'], 0.5, event['description'], rotation=90)
```

#### 6.3.2 因果路径分解

**分解突变效应的因果机制**:
```
突变 R83A 的因果路径:

1. 序列变化: R → A
   ├─ 直接效应: 失去正电荷（30%）
   └─ 结构效应: 侧链变短（10%）

2. 结构变化: 局部构象改变
   ├─ 氢键断裂: 2个氢键消失（25%）
   └─ 疏水核心松弛（15%）

3. 界面变化: 接触面积减少
   └─ 结合亲和力下降（20%）

总效应: ΔΔG = +3.2 kcal/mol
```

#### 6.3.3 进化证据权重

**对于每个PPI预测，输出进化支持证据**:
```
PPI: ProteinA - ProteinB
预测: 存在 (概率 0.92)

进化证据:
  - 在12/15个物种中保守 (80%)
  - 共进化信号强度: 0.85
  - 祖先重建: 很可能在LUCA就存在
  - 选择压力: 强负选择 (dN/dS = 0.15)
  
因果证据:
  - 10个已知突变破坏PPI (100%一致)
  - 药物X抑制PPI (文献支持)
  
置信度: 高
```

### 6.4 实际应用案例

#### 案例1: 指导蛋白质工程

**任务**: 优化酶-底物PPI以提高催化效率

```python
# 搜索最佳突变组合
def optimize_ppi_strength(enzyme, substrate, target_strength=1.5):
    """
    寻找突变组合，使PPI强度增加到target_strength倍
    
    约束:
    1. 保持酶的稳定性
    2. 突变在进化上可行
    3. 最少突变数
    """
    
    current_strength = model.predict_ppi(enzyme, substrate)
    
    # 反事实搜索
    best_mutations = model.counterfactual_search(
        enzyme, substrate,
        target_ppi=current_strength * target_strength,
        constraints={
            'max_mutations': 5,
            'maintain_stability': True,
            'evolutionary_plausibility': 0.6  # 至少60%可行性
        }
    )
    
    return best_mutations

# 示例输出
"""
建议突变组合:
  1. A_T45S (效应: +0.3, 可行性: 0.8)
  2. A_R67K (效应: +0.2, 可行性: 0.9)  # 保守替换
  3. B_E123D (效应: +0.15, 可行性: 0.85)

预测总效应: PPI强度 × 1.52
进化支持: 所有突变在自然界中均有观察
实验优先级: 高
"""
```

#### 案例2: 解释疾病突变

**任务**: 解释致病突变为何破坏PPI

```python
# 分析致病突变
def explain_pathogenic_mutation(protein_A, protein_B, mutation):
    """
    解释突变如何导致疾病
    """
    
    # 1. 预测效应
    effect = model.predict_mutation_effect(protein_A, protein_B, mutation)
    
    # 2. 因果分解
    mechanism = effect['causal_mechanism']
    
    # 3. 进化证据
    conservation = model.evolutionary_conservation(protein_A, mutation.position)
    
    # 4. 生成报告
    report = f"""
    突变: {mutation}
    
    预测效应: PPI强度下降 {effect['delta_ppi']:.2f} ({effect['confidence_interval']})
    
    致病机制:
      1. 破坏关键盐桥 ({mechanism['electrostatic']:.1%})
      2. 降低蛋白稳定性 ({mechanism['stability']:.1%})
      3. 改变界面几何 ({mechanism['interface']:.1%})
    
    进化证据:
      - 该位点在脊椎动物中100%保守
      - dN/dS = 0.05 (强负选择)
      - 自然界中从未观察到类似突变
    
    结论: 高置信度致病突变
    """
    
    return report
```

#### 案例3: 药物靶点发现

**任务**: 寻找可被小分子破坏的PPI

```python
def identify_druggable_ppi(ppi_network):
    """
    识别可被药物干预的PPI
    
    标准:
    1. PPI破坏会产生治疗效果
    2. 界面有可成药的口袋
    3. 不会破坏其他关键PPI（选择性）
    """
    
    druggable_ppis = []
    
    for ppi in ppi_network:
        # 1. 预测破坏PPI的效应
        effect = model.predict_intervention_effect(
            ppi, intervention_type='inhibition'
        )
        
        # 2. 评估可成药性
        druggability = assess_druggability(ppi.interface)
        
        # 3. 评估选择性（因果推理）
        off_targets = model.predict_off_target_effects(ppi)
        
        if effect.is_beneficial() and druggability > 0.7 and len(off_targets) < 3:
            druggable_ppis.append({
                'ppi': ppi,
                'therapeutic_effect': effect,
                'druggability_score': druggability,
                'selectivity': 1 / (1 + len(off_targets))
            })
    
    return druggable_ppis
```

---

## 7. 预期结果与影响

### 7.1 性能预期

#### 7.1.1 定量结果

| 任务 | 指标 | 当前SOTA | 预期结果 | 提升 |
|------|------|---------|---------|------|
| Human PPI预测 | AUPR | 0.75 | 0.78 | +4% |
| 跨物种泛化 | AUPR | 0.45 | **0.65** | **+44%** ⭐ |
| 突变效应预测 | Spearman ρ | 0.52 | **0.71** | **+37%** ⭐ |
| 反事实推理 | N/A | - | **首次实现** | - |

**关键优势**: 在分布外任务上远超现有方法

#### 7.1.2 定性结果

**新能力**（现有方法无法做到）:
1. ✅ 预测PPI的进化起源和历史
2. ✅ 解释为什么某些PPI存在/不存在（因果机制）
3. ✅ 预测反事实场景（"如果...会怎样"）
4. ✅ 指导蛋白质工程（逆向设计突变）
5. ✅ 量化预测的不确定性（进化随机性）

### 7.2 生物学发现

#### 预期新发现

**发现1: PPI的进化约束模式**
- 识别"进化上不可或缺"的PPI（强负选择）
- 发现"进化上灵活"的PPI（可塑性高）
- 绘制PPI网络的"进化地图"

**发现2: 共进化网络模块**
- 哪些蛋白质家族倾向于共进化？
- 共进化是否预测功能耦合？
- 跨家族的共进化模式

**发现3: 物种特异的PPI适应**
- 为什么某些PPI只在特定物种存在？
- 环境适应 vs 中性漂变的相对贡献
- 物种特异PPI的功能意义

**发现4: 致病突变的因果图谱**
- 哪些因果路径最常被致病突变破坏？
- 是否存在"脆弱路径"（易受突变影响）？
- 补偿机制：自然界如何缓冲突变效应？

### 7.3 方法论影响

#### 7.3.1 对生物信息学的影响

**新范式**: 从"预测关联"到"理解因果"
- 启发其他任务（基因调控、代谢网络）采用因果框架
- 推动"进化深度学习"成为独立领域
- 证明观察性生物数据可用于因果推断

#### 7.3.2 对AI/ML的影响

**技术贡献**:
- 利用时间序列数据（进化）进行因果发现的新方法
- 结构因果模型与深度学习的融合范式
- 混合监督信号（进化约束 + 干预实验）的训练策略

#### 7.3.3 对药物发现的影响

**实际应用**:
- 减少湿实验成本（预测突变效应）
- 加速蛋白质工程迭代
- 精准医疗：个性化预测突变影响

### 7.4 论文发表策略

#### Nature主刊 (首选)

**优势**:
- ✅ 理论深度：进化生物学 + 因果推断 + 深度学习
- ✅ 广泛影响：可推广到其他生物网络
- ✅ 实际应用：药物、疾病、合成生物学
- ✅ 跨学科：吸引多领域读者

**Story Arc**:
```
Title: "Causal Inference from Evolutionary Dynamics Enables 
        Predictive and Interpretable Protein Interaction Modeling"

Abstract: 
  - 动机：PPI预测缺乏因果理解
  - 洞察：进化 = 自然的长期干预实验
  - 方法：联合建模进化轨迹和因果机制
  - 结果：跨物种泛化+44%，突变预测+37%，首次实现反事实推理
  - 影响：发现新生物学规律，指导蛋白工程

Main Figures:
  1. 框架总览（进化+因果的联合模型）
  2. 跨物种泛化性能（显著优势）
  3. 突变效应预测（因果路径分解）
  4. 反事实案例研究（蛋白工程应用）
  5. 新生物学发现（进化约束图谱）
```

#### 备选期刊

- **Nature Methods**: 如果更强调方法论
- **Nature Biotechnology**: 如果更强调应用
- **PNAS**: 理论+应用都强
- **JMLR / ICML**: 如果强调机器学习创新

### 7.5 开源与社区影响

**开源计划**:
```
GitHub Repo: evolutionary-causal-ppi
├── data/                   # 数据收集脚本
├── models/                 # 模型实现
│   ├── evolution.py        # 进化模块
│   ├── causal.py          # 因果模块
│   └── intervention.py    # 干预预测
├── experiments/           # 实验复现脚本
├── tutorials/             # 使用教程
└── pretrained/            # 预训练模型
```

**社区工具**:
- Web服务器：输入蛋白质对 → 输出预测 + 解释
- Colab Notebook：交互式演示
- API：集成到其他工具（AlphaFold, Rosetta）

---

## 8. 实施路线图

### 8.1 时间线（12个月）

#### 阶段1: 数据准备与基础模型 (Month 1-3)

**Month 1: 数据收集**
- [ ] 收集多物种PPI数据（至少10个物种）
- [ ] 构建物种树（分支长度标定）
- [ ] 整合干预实验数据（SKEMPI等）
- [ ] 数据预处理和质控

**Month 2: 基础模型实现**
- [ ] 实现ESM-2编码器
- [ ] 实现祖先序列重建网络
- [ ] 实现神经ODE进化模型
- [ ] 基线PPI预测器

**Month 3: 初步验证**
- [ ] 在PRING数据上测试基础模型
- [ ] 验证祖先重建的质量
- [ ] 调试和优化

**里程碑1**: 基础PPI预测性能达到SOTA水平

#### 阶段2: 因果模型开发 (Month 4-6)

**Month 4: 因果图学习**
- [ ] 实现进化因果发现算法
- [ ] 从多物种数据学习因果图
- [ ] 验证因果关系的合理性

**Month 5: 结构因果模型**
- [ ] 实现SCM（序列→结构→界面→PPI）
- [ ] 实现干预操作
- [ ] 实现反事实推理

**Month 6: 整合训练**
- [ ] 联合训练进化模块和因果模块
- [ ] 多任务学习策略
- [ ] 超参数调优

**里程碑2**: 因果模型能正确预测已知干预效应

#### 阶段3: 全面评估 (Month 7-9)

**Month 7: 标准任务评估**
- [ ] PRING benchmark评估
- [ ] 跨物种泛化测试
- [ ] 与所有基线对比

**Month 8: 干预任务评估**
- [ ] SKEMPI突变效应预测
- [ ] 反事实推理案例研究
- [ ] 消融实验

**Month 9: 可解释性分析**
- [ ] 进化轨迹可视化
- [ ] 因果路径分解
- [ ] 生物学验证（文献）

**里程碑3**: 所有评估完成，结果达到预期

#### 阶段4: 论文撰写与应用开发 (Month 10-12)

**Month 10: 论文初稿**
- [ ] 撰写方法部分
- [ ] 制作所有图表
- [ ] 初稿内部审阅

**Month 11: 实际应用案例**
- [ ] 蛋白质工程案例
- [ ] 疾病突变解释案例
- [ ] 药物靶点发现案例

**Month 12: 投稿准备**
- [ ] 论文修改和润色
- [ ] 补充材料准备
- [ ] 代码整理和开源准备
- [ ] 投稿

**里程碑4**: 论文投稿

### 8.2 人力需求

**核心团队**:
- **1名 PI/高级研究员**: 整体规划和理论指导
- **2名 博士生/博后**:
  - 1名负责进化模型（生物信息学背景）
  - 1名负责因果模型（机器学习背景）
- **1名 研究助理**: 数据收集和预处理

**协作人员**:
- **结构生物学专家**: 验证结构预测
- **进化生物学专家**: 验证进化推断
- **计算资源支持**: GPU集群管理

### 8.3 计算资源需求

**训练**:
- **GPU**: 4-8× A100 (80GB)
- **训练时间**: ~2-3周
- **存储**: ~10TB（多物种数据 + 模型checkpoints）

**推理**:
- **GPU**: 1× V100即可
- **Web服务**: 轻量级部署

**预算估算**:
- 云计算（AWS/GCP）: ~$20,000
- 或使用学校/机构GPU集群

### 8.4 风险与应对

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| 多物种数据质量差 | 中 | 高 | 使用高置信度数据源；数据清洗流程 |
| 祖先重建不准确 | 高 | 中 | 采用分布估计而非点估计；不确定性量化 |
| 因果图学习困难 | 中 | 高 | 从简单的手工因果图开始；逐步放松假设 |
| 计算资源不足 | 低 | 中 | 申请GPU资源；优化模型规模 |
| 审稿人质疑因果推断 | 中 | 高 | 准备严格的统计验证；使用已知干预数据验证 |

---

## 9. 潜在挑战与解决方案

### 9.1 技术挑战

#### 挑战1: 祖先序列重建的不确定性

**问题**: 祖先序列无法直接观察，重建存在误差

**解决方案**:
1. **贝叶斯方法**: 输出祖先序列的后验分布，而非单点估计
2. **敏感性分析**: 测试不同祖先假设下的结果稳定性
3. **保守策略**: 只使用高置信度的祖先推断

```python
# 代码示例
ancestral_dist = model.sample_ancestors(modern_seqs, n_samples=1000)
predictions = [model.predict_ppi(anc_A, anc_B) for anc_A, anc_B in ancestral_dist]
mean_pred = np.mean(predictions)
uncertainty = np.std(predictions)

if uncertainty > threshold:
    print("Warning: High uncertainty in ancestral reconstruction")
```

#### 挑战2: 因果图的识别性 (Identifiability)

**问题**: 从观察数据推断因果图是欠定问题（multiple DAGs可以拟合同样数据）

**解决方案**:
1. **引入生物学先验**: 序列必定在结构之前，结构必定在PPI之前
2. **利用干预数据**: 干预实验可以排除等价DAG
3. **时间顺序**: 进化数据天然提供时间因果（祖先→后代）

```python
# 引入生物学先验
class BiologicallyConstrainedDAG:
    def __init__(self):
        self.mandatory_edges = [
            ('sequence_A', 'structure_A'),
            ('structure_A', 'interface'),
            ('interface', 'ppi')
        ]
        self.forbidden_edges = [
            ('ppi', 'sequence_A'),  # PPI不能导致序列变化
        ]
    
    def is_valid_dag(self, dag):
        # 检查是否满足生物学约束
        for edge in self.mandatory_edges:
            if edge not in dag.edges():
                return False
        for edge in self.forbidden_edges:
            if edge in dag.edges():
                return False
        return True
```

#### 挑战3: 计算复杂度

**问题**: 
- 多物种 × 多时间点 → 组合爆炸
- 反事实推理需要多次正向传播

**解决方案**:
1. **分层采样**: 不需要枚举所有祖先组合，采样代表性样本
2. **缓存**: 缓存中间计算结果（如结构预测）
3. **近似推断**: 使用变分推断代替精确推断

```python
# 使用缓存减少重复计算
from functools import lru_cache

@lru_cache(maxsize=10000)
def predict_structure(sequence):
    # 结构预测很慢，缓存结果
    return alphafold.predict(sequence)

# 变分近似
class VariationalCausalInference:
    def approximate_counterfactual(self, protein_A, protein_B, intervention):
        # 不精确求解SCM，而是用变分方法近似
        q = self.variational_posterior(protein_A, protein_B)
        return self.monte_carlo_estimate(q, intervention)
```

### 9.2 数据挑战

#### 挑战4: 多物种PPI数据的异质性

**问题**:
- 不同物种的PPI检测方法不同（Y2H, co-IP, etc.）
- 数据完整度差异大（human > yeast >> 其他）
- 假阳性率不同

**解决方案**:
1. **置信度加权**: 根据实验方法和数据来源分配权重
2. **迁移学习**: 从数据丰富的物种迁移到数据稀疏的物种
3. **数据增强**: 利用直系同源关系扩充训练数据

```python
class MultiSpeciesDataLoader:
    def __init__(self):
        self.confidence_weights = {
            'human': 1.0,      # 高质量
            'yeast': 0.9,      # 高质量
            'arath': 0.7,      # 中等质量
            'ecoli': 0.6       # 中等质量
        }
    
    def weighted_loss(self, predictions, labels, species):
        weight = self.confidence_weights[species]
        loss = F.binary_cross_entropy(predictions, labels, reduction='none')
        return (loss * weight).mean()
```

#### 挑战5: 干预数据的稀缺性

**问题**: SKEMPI等数据库规模有限（<10K突变）

**解决方案**:
1. **半监督学习**: 大量无干预数据 + 少量干预数据
2. **主动学习**: 优先实验验证模型不确定的突变
3. **模拟数据**: 使用物理模拟（FoldX, Rosetta）生成伪标签

```python
# 半监督学习
def semi_supervised_loss(self, batch):
    if batch.has_intervention_label:
        # 有监督：直接匹配干预效应
        loss_supervised = F.mse_loss(
            self.predict_intervention(batch), 
            batch.true_effect
        )
        return loss_supervised
    else:
        # 无监督：进化一致性
        loss_unsupervised = self.evolution_consistency_loss(batch)
        return 0.1 * loss_unsupervised  # 降低权重
```

### 9.3 生物学挑战

#### 挑战6: 验证生物学发现

**问题**: 如何证明发现的规律是真实的生物学现象，而非统计假象？

**解决方案**:
1. **文献验证**: 与已知生物学知识对照
2. **功能富集**: 发现的模式是否有GO term富集？
3. **协作验证**: 与实验室合作进行湿实验验证

```python
# 自动文献验证
class LiteratureValidator:
    def __init__(self):
        self.pubmed = PubMedAPI()
    
    def validate_finding(self, protein_A, protein_B, predicted_mechanism):
        # 搜索PubMed
        query = f"{protein_A} AND {protein_B} AND ({predicted_mechanism})"
        papers = self.pubmed.search(query, max_results=50)
        
        # 提取支持/反对证据
        support = self.extract_evidence(papers, predicted_mechanism)
        
        return {
            'literature_support': len(support) > 0,
            'supporting_papers': support,
            'confidence': len(support) / len(papers) if papers else 0
        }
```

#### 挑战7: 进化模型的简化假设

**问题**: 
- 假设中性进化 + 选择，忽略了重组、基因转移等
- 分子钟假设可能不成立

**解决方案**:
1. **松弛假设**: 允许不同支系有不同进化速率
2. **模型选择**: 测试多个进化模型（JC69, GTR, etc.）
3. **稳健性分析**: 结果是否对模型假设敏感？

```python
# 测试多个进化模型
def robust_ancestral_reconstruction(modern_seqs, tree):
    results = []
    for model in ['JC69', 'GTR', 'Neural']:
        ancestors = reconstruct_ancestors(modern_seqs, tree, model=model)
        results.append(ancestors)
    
    # 比较不同模型的一致性
    consistency = compute_consensus(results)
    
    if consistency < 0.8:
        warnings.warn("Low consistency across evolution models")
    
    return results[0]  # 返回最复杂模型的结果
```

### 9.4 审稿挑战

#### 潜在审稿意见与回应

**意见1**: "因果推断需要随机对照试验，观察性数据不能推断因果"

**回应**:
- Pearl的因果阶梯理论已证明，在满足特定假设下（可忽略性、一致性），观察性数据可以推断因果
- 进化数据提供了"自然实验"（遗传漂变 = 随机化）
- 我们使用干预实验数据（SKEMPI）作为因果发现的验证
- 引用文献：Pearl (2009), Hernán & Robins (2020)

**意见2**: "祖先重建误差会传播到下游预测"

**回应**:
- 我们使用贝叶斯方法量化祖先重建的不确定性
- 敏感性分析显示，结果对祖先假设稳健
- 即使祖先重建有误差，进化一致性约束仍然有效（补充图S5）

**意见3**: "多物种数据质量参差不齐"

**回应**:
- 我们采用置信度加权策略
- 主要结果在高质量数据（human, yeast）上可复现
- 消融实验显示，即使只用5个高质量物种，仍有显著提升

**意见4**: "计算成本太高，难以推广"

**回应**:
- 训练成本高（一次性），但推理成本低
- 提供预训练模型，下游应用无需重新训练
- 对于关键应用（药物设计、疾病诊断），计算成本可接受

---

## 10. 总结与展望

### 10.1 核心贡献总结

**理论贡献**:
1. 首次提出PPI的**进化动力学**与**因果推断**统一框架
2. 证明可以从观察性进化数据推断因果关系
3. 建立PPI的**结构因果模型**（SCM）

**方法贡献**:
1. 深度学习驱动的祖先序列重建（联合优化）
2. 进化驱动的因果图学习算法
3. 反事实推理用于蛋白质工程

**应用贡献**:
1. 跨物种PPI预测（泛化能力提升44%）
2. 突变效应预测（准确率提升37%）
3. 蛋白质设计的因果指导

### 10.2 对领域的长期影响

**生物信息学**:
- 开创"进化深度学习"新方向
- 推动因果推断在生物数据中的应用
- 提供可解释AI的范例

**计算生物学**:
- 改变PPI预测的范式（从关联到因果）
- 启发其他生物网络（代谢、调控）采用类似框架
- 促进实验-计算的闭环

**药物发现**:
- 加速蛋白质工程（预测-设计-验证）
- 精准医疗（个性化突变效应预测）
- 药物靶点发现（可干预PPI识别）

### 10.3 未来研究方向

**短期（1-2年）**:
1. 扩展到更多物种（覆盖生命之树）
2. 整合结构信息（AlphaFold预测）
3. 开发用户友好的Web工具

**中期（3-5年）**:
1. 扩展到其他生物网络（基因调控、代谢）
2. 多模态整合（序列+结构+功能）
3. 与实验室合作进行大规模验证

**长期（5+年）**:
1. "逆向工程"生命系统：从功能需求推断序列
2. 合成生物学：设计全新PPI网络
3. 进化深度学习的通用框架

---

## 11. 参考文献（部分）

### 进化生物学
1. Yang, Z. (2007). PAML: Phylogenetic analysis by maximum likelihood. *Molecular Biology and Evolution*.
2. Thornton, J. W. (2004). Resurrecting ancient genes. *Nature Reviews Genetics*.

### 因果推断  
3. Pearl, J. (2009). *Causality: Models, Reasoning and Inference*. Cambridge University Press.
4. Peters, J., et al. (2017). *Elements of Causal Inference*. MIT Press.

### 蛋白质相互作用
5. Rao, V. S., et al. (2014). Protein-protein interaction detection. *Proteomics*.
6. Browne, F., et al. (2019). Computational prediction of protein-protein interactions. *Bioinformatics*.

### 深度学习 + 生物学
7. Jumper, J., et al. (2021). AlphaFold 2. *Nature*.
8. Rives, A., et al. (2021). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. *PNAS*.

### 因果推断 + ML
9. Schölkopf, B., et al. (2021). Toward causal representation learning. *Proceedings of the IEEE*.
10. Kaddour, J., et al. (2022). Causal machine learning: A survey. *arXiv*.

---

**文档版本**: 1.0  
**最后更新**: 2025年10月  
**作者**: Hyperbolic_Big项目组  
**联系方式**: [待添加]

---

## 附录

### A. 术语表

- **PPI**: Protein-Protein Interaction（蛋白质相互作用）
- **SCM**: Structural Causal Model（结构因果模型）
- **DAG**: Directed Acyclic Graph（有向无环图）
- **ASR**: Ancestral Sequence Reconstruction（祖先序列重建）
- **ODE**: Ordinary Differential Equation（常微分方程）
- **VAE**: Variational Autoencoder（变分自编码器）

### B. 数学符号

- $P_{AB}(t)$: 时间$t$时蛋白A和B的相互作用概率
- $S_A(t)$: 时间$t$时蛋白A的序列状态
- $\theta_{\text{evo}}$: 进化过程参数
- $\theta_{\text{sel}}$: 选择压力参数
- $do(X)$: Pearl的干预算子

### C. 代码仓库结构

```
evolutionary-causal-ppi/
├── README.md
├── requirements.txt
├── setup.py
├── data/
│   ├── collectors/          # 数据收集脚本
│   ├── preprocessors/       # 数据预处理
│   └── loaders/            # DataLoader
├── models/
│   ├── evolution/          # 进化模块
│   │   ├── ancestral.py
│   │   ├── neural_ode.py
│   │   └── coevolution.py
│   ├── causal/            # 因果模块
│   │   ├── discovery.py
│   │   ├── scm.py
│   │   └── intervention.py
│   └── encoders/          # 编码器
│       ├── esm.py
│       └── tree_lstm.py
├── training/
│   ├── trainers.py
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── visualizations.py
├── experiments/
│   ├── baselines/
│   ├── ablations/
│   └── case_studies/
├── notebooks/
│   └── tutorials/
└── tests/
    └── unit_tests/
```

---

**致谢**

感谢PRING团队提供的高质量基准数据集。感谢ESM团队开源的蛋白质语言模型。

---

**End of Document**
