#!/usr/bin/env python3
"""
聚类分析器
分析STRING聚类数据，为MoE专家模型提供分组依据
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging
from typing import Dict, List, Set, Tuple
import networkx as nx
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class ClusterAnalyzer:
    """STRING聚类数据分析器"""
    
    def __init__(self, data_dir: str = "data"):
        """初始化聚类分析器"""
        self.data_dir = Path(data_dir)
        
        # 聚类相关文件
        self.cluster_info_file = self.data_dir / "clusters.info.v12.0.txt.gz"
        self.cluster_proteins_file = self.data_dir / "clusters.proteins.v12.0.txt.gz"
        self.cluster_tree_file = self.data_dir / "clusters.tree.v12.0.txt.gz"
        
        # 缓存数据
        self.cluster_info = None
        self.protein_clusters = None
        self.cluster_tree = None
    
    def load_cluster_info(self) -> pd.DataFrame:
        """加载聚类信息"""
        if self.cluster_info is not None:
            return self.cluster_info
        
        logger.info("加载聚类信息...")
        
        try:
            cluster_data = []
            with gzip.open(self.cluster_info_file, 'rt') as f:
                # 跳过头部
                header = f.readline().strip().split('\t')
                logger.info(f"聚类信息列: {header}")
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        cluster_data.append({
                            'cluster_id': parts[0],
                            'cluster_name': parts[1] if len(parts) > 1 else '',
                            'cluster_description': parts[2] if len(parts) > 2 else '',
                            'cluster_size': int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
                        })
            
            self.cluster_info = pd.DataFrame(cluster_data)
            logger.info(f"加载了 {len(self.cluster_info)} 个聚类信息")
            
        except FileNotFoundError:
            logger.warning(f"聚类信息文件不存在: {self.cluster_info_file}")
            self.cluster_info = pd.DataFrame()
        
        return self.cluster_info
    
    def load_protein_clusters(self) -> pd.DataFrame:
        """加载蛋白质-聚类映射"""
        if self.protein_clusters is not None:
            return self.protein_clusters
        
        logger.info("加载蛋白质-聚类映射...")
        
        try:
            mapping_data = []
            with gzip.open(self.cluster_proteins_file, 'rt') as f:
                # 跳过头部
                header = f.readline().strip().split('\t')
                logger.info(f"蛋白质聚类映射列: {header}")
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        mapping_data.append({
                            'cluster_id': parts[0],
                            'protein_id': parts[1]
                        })
            
            self.protein_clusters = pd.DataFrame(mapping_data)
            logger.info(f"加载了 {len(self.protein_clusters)} 个蛋白质-聚类映射")
            
        except FileNotFoundError:
            logger.warning(f"蛋白质聚类映射文件不存在: {self.cluster_proteins_file}")
            self.protein_clusters = pd.DataFrame()
        
        return self.protein_clusters
    
    def load_cluster_tree(self) -> pd.DataFrame:
        """加载聚类层次树"""
        if self.cluster_tree is not None:
            return self.cluster_tree
        
        logger.info("加载聚类层次树...")
        
        try:
            tree_data = []
            with gzip.open(self.cluster_tree_file, 'rt') as f:
                # 跳过头部
                header = f.readline().strip().split('\t')
                logger.info(f"聚类树列: {header}")
                
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        tree_data.append({
                            'child_cluster': parts[0],
                            'parent_cluster': parts[1],
                            'distance': float(parts[2]) if parts[2].replace('.', '').isdigit() else 0.0
                        })
            
            self.cluster_tree = pd.DataFrame(tree_data)
            logger.info(f"加载了 {len(self.cluster_tree)} 个聚类层次关系")
            
        except FileNotFoundError:
            logger.warning(f"聚类树文件不存在: {self.cluster_tree_file}")
            self.cluster_tree = pd.DataFrame()
        
        return self.cluster_tree
    
    def analyze_cluster_statistics(self) -> Dict:
        """分析聚类统计信息"""
        logger.info("分析聚类统计信息...")
        
        cluster_info = self.load_cluster_info()
        protein_clusters = self.load_protein_clusters()
        
        if cluster_info.empty or protein_clusters.empty:
            logger.warning("聚类数据不完整，跳过统计分析")
            return {}
        
        # 基础统计
        stats = {
            'total_clusters': len(cluster_info),
            'total_protein_cluster_mappings': len(protein_clusters),
            'proteins_with_clusters': protein_clusters['protein_id'].nunique(),
            'clusters_with_proteins': protein_clusters['cluster_id'].nunique()
        }
        
        # 聚类大小分布
        cluster_sizes = protein_clusters['cluster_id'].value_counts()
        stats['cluster_size_distribution'] = {
            'mean': cluster_sizes.mean(),
            'std': cluster_sizes.std(),
            'min': cluster_sizes.min(),
            'max': cluster_sizes.max(),
            'percentiles': cluster_sizes.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        }
        
        # 大聚类分析（前20个）
        top_clusters = cluster_sizes.head(20)
        stats['top_clusters'] = top_clusters.to_dict()
        
        # 蛋白质每个聚类的分布
        proteins_per_cluster_dist = protein_clusters.groupby('protein_id')['cluster_id'].count()
        stats['proteins_cluster_membership'] = {
            'single_cluster': (proteins_per_cluster_dist == 1).sum(),
            'multiple_clusters': (proteins_per_cluster_dist > 1).sum(),
            'max_clusters_per_protein': proteins_per_cluster_dist.max(),
            'avg_clusters_per_protein': proteins_per_cluster_dist.mean()
        }
        
        logger.info("聚类统计结果:")
        logger.info(f"  总聚类数: {stats['total_clusters']:,}")
        logger.info(f"  有聚类的蛋白质数: {stats['proteins_with_clusters']:,}")
        logger.info(f"  平均聚类大小: {stats['cluster_size_distribution']['mean']:.1f}")
        logger.info(f"  最大聚类大小: {stats['cluster_size_distribution']['max']:,}")
        logger.info(f"  蛋白质平均所属聚类数: {stats['proteins_cluster_membership']['avg_clusters_per_protein']:.2f}")
        
        return stats
    
    def get_moe_expert_groups(self, min_cluster_size: int = 100, 
                             max_num_experts: int = 50) -> Dict[str, List[str]]:
        """为MoE模型创建专家分组"""
        logger.info(f"创建MoE专家分组，最小聚类大小: {min_cluster_size}, 最大专家数: {max_num_experts}")
        
        protein_clusters = self.load_protein_clusters()
        
        if protein_clusters.empty:
            logger.warning("无聚类数据，使用随机分组策略")
            return self._create_random_expert_groups(max_num_experts)
        
        # 统计每个聚类的大小
        cluster_sizes = protein_clusters['cluster_id'].value_counts()
        
        # 选择大小合适的聚类作为专家
        suitable_clusters = cluster_sizes[cluster_sizes >= min_cluster_size].head(max_num_experts)
        
        expert_groups = {}
        for cluster_id, size in suitable_clusters.items():
            # 获取该聚类中的所有蛋白质
            proteins_in_cluster = protein_clusters[
                protein_clusters['cluster_id'] == cluster_id
            ]['protein_id'].tolist()
            
            expert_groups[f"expert_{cluster_id}"] = proteins_in_cluster
        
        # 为未分组的蛋白质创建默认专家
        all_clustered_proteins = set()
        for proteins in expert_groups.values():
            all_clustered_proteins.update(proteins)
        
        all_proteins = set(protein_clusters['protein_id'].unique())
        unclustered_proteins = all_proteins - all_clustered_proteins
        
        if unclustered_proteins:
            expert_groups["expert_others"] = list(unclustered_proteins)
        
        logger.info(f"创建了 {len(expert_groups)} 个专家组:")
        for expert_name, proteins in expert_groups.items():
            logger.info(f"  {expert_name}: {len(proteins):,} 个蛋白质")
        
        return expert_groups
    
    def _create_random_expert_groups(self, num_experts: int, 
                                   protein_list: List[str] = None) -> Dict[str, List[str]]:
        """创建随机专家分组（聚类数据不可用时的备选方案）"""
        logger.info(f"创建 {num_experts} 个随机专家分组")
        
        if protein_list is None:
            # 如果没有提供蛋白质列表，创建空的分组结构
            return {f"expert_{i}": [] for i in range(num_experts)}
        
        # 随机分配蛋白质到专家组
        np.random.shuffle(protein_list)
        proteins_per_expert = len(protein_list) // num_experts
        
        expert_groups = {}
        for i in range(num_experts):
            start_idx = i * proteins_per_expert
            end_idx = start_idx + proteins_per_expert if i < num_experts - 1 else len(protein_list)
            expert_groups[f"expert_{i}"] = protein_list[start_idx:end_idx]
        
        return expert_groups
    
    def analyze_cluster_hierarchy(self) -> Dict:
        """分析聚类层次结构"""
        logger.info("分析聚类层次结构...")
        
        cluster_tree = self.load_cluster_tree()
        
        if cluster_tree.empty:
            logger.warning("无聚类树数据")
            return {}
        
        # 构建层次图
        G = nx.DiGraph()
        for _, row in cluster_tree.iterrows():
            G.add_edge(row['parent_cluster'], row['child_cluster'], 
                      distance=row['distance'])
        
        # 分析层次结构
        hierarchy_stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'max_depth': 0,
            'num_roots': 0,
            'num_leaves': 0
        }
        
        # 找到根节点（没有入边的节点）
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        hierarchy_stats['num_roots'] = len(roots)
        
        # 找到叶节点（没有出边的节点）
        leaves = [node for node in G.nodes() if G.out_degree(node) == 0]
        hierarchy_stats['num_leaves'] = len(leaves)
        
        # 计算最大深度
        if roots:
            max_depth = 0
            for root in roots:
                try:
                    depths = nx.single_source_shortest_path_length(G, root)
                    max_depth = max(max_depth, max(depths.values()) if depths else 0)
                except:
                    continue
            hierarchy_stats['max_depth'] = max_depth
        
        logger.info("聚类层次结构分析结果:")
        logger.info(f"  节点数: {hierarchy_stats['total_nodes']:,}")
        logger.info(f"  边数: {hierarchy_stats['total_edges']:,}")
        logger.info(f"  根节点数: {hierarchy_stats['num_roots']:,}")
        logger.info(f"  叶节点数: {hierarchy_stats['num_leaves']:,}")
        logger.info(f"  最大深度: {hierarchy_stats['max_depth']}")
        
        return hierarchy_stats
    
    def get_hierarchical_expert_groups(self, target_depth: int = 2) -> Dict[str, List[str]]:
        """基于层次结构创建专家分组"""
        logger.info(f"基于层次结构创建专家分组，目标深度: {target_depth}")
        
        cluster_tree = self.load_cluster_tree()
        protein_clusters = self.load_protein_clusters()
        
        if cluster_tree.empty or protein_clusters.empty:
            logger.warning("缺少层次数据，使用平铺聚类方案")
            return self.get_moe_expert_groups()
        
        # 构建层次图
        G = nx.DiGraph()
        for _, row in cluster_tree.iterrows():
            G.add_edge(row['parent_cluster'], row['child_cluster'], 
                      distance=row['distance'])
        
        # 找到指定深度的节点作为专家组
        roots = [node for node in G.nodes() if G.in_degree(node) == 0]
        expert_clusters = set()
        
        for root in roots:
            try:
                # 获取指定深度的节点
                paths = nx.single_source_shortest_path(G, root, cutoff=target_depth)
                for path in paths.values():
                    if len(path) == target_depth + 1:  # 路径长度=深度+1
                        expert_clusters.add(path[-1])
            except:
                continue
        
        # 为每个专家聚类获取蛋白质列表
        expert_groups = {}
        for cluster_id in expert_clusters:
            proteins = protein_clusters[
                protein_clusters['cluster_id'] == cluster_id
            ]['protein_id'].tolist()
            
            if proteins:  # 只保留非空的专家组
                expert_groups[f"expert_hierarchical_{cluster_id}"] = proteins
        
        logger.info(f"基于层次结构创建了 {len(expert_groups)} 个专家组")
        
        return expert_groups
    
    def validate_expert_groups(self, expert_groups: Dict[str, List[str]], 
                             filtered_proteins: Set[str]) -> Dict[str, List[str]]:
        """验证并过滤专家分组，只保留在过滤后蛋白质列表中的蛋白质"""
        logger.info("验证专家分组...")
        
        validated_groups = {}
        total_proteins_before = sum(len(proteins) for proteins in expert_groups.values())
        total_proteins_after = 0
        
        for expert_name, proteins in expert_groups.items():
            # 只保留在过滤后蛋白质列表中的蛋白质
            valid_proteins = [p for p in proteins if p in filtered_proteins]
            
            if valid_proteins:  # 只保留非空的专家组
                validated_groups[expert_name] = valid_proteins
                total_proteins_after += len(valid_proteins)
        
        logger.info(f"专家分组验证结果:")
        logger.info(f"  验证前专家组数: {len(expert_groups)}")
        logger.info(f"  验证后专家组数: {len(validated_groups)}")
        logger.info(f"  验证前蛋白质总数: {total_proteins_before:,}")
        logger.info(f"  验证后蛋白质总数: {total_proteins_after:,}")
        
        return validated_groups