"""
PRING数据集PyTorch实现

提供两种Dataset:
1. PRINGPairDataset - 用于成对PPI预测（二分类）
2. PRINGGraphDataset - 用于图重建任务
"""

import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from Bio import SeqIO

import torch
from torch.utils.data import Dataset, DataLoader

from .config import PRINGConfig

logger = logging.getLogger(__name__)


class PRINGPairDataset(Dataset):
    """
    PRING成对PPI数据集
    
    用于二分类任务：给定两个蛋白质序列，预测是否存在相互作用
    """
    
    def __init__(
        self,
        config: PRINGConfig,
        transform: Optional[Callable] = None,
        max_length: int = 1000,
        return_ids: bool = False
    ):
        """
        初始化数据集
        
        Args:
            config: PRING配置对象
            transform: 序列转换函数（如tokenization）
            max_length: 序列最大长度
            return_ids: 是否返回蛋白质ID
        """
        self.config = config
        self.transform = transform
        self.max_length = max_length
        self.return_ids = return_ids
        
        # 验证配置
        if not config.validate():
            raise FileNotFoundError("配置验证失败，请检查文件路径")
        
        # 加载数据
        logger.info(f"加载PRING数据: {config}")
        self._load_ppi_pairs()
        self._load_sequences()
        
        logger.info(f"数据加载完成: {len(self)} 个PPI对, {len(self.sequences)} 个蛋白质序列")
    
    def _load_ppi_pairs(self):
        """加载PPI对"""
        logger.info(f"加载PPI文件: {self.config.ppi_file}")
        
        # 读取PPI文件（使用\t作为分隔符）
        self.ppi_df = pd.read_csv(
            self.config.ppi_file,
            sep='\t',
            names=['protein1', 'protein2', 'label'],
            dtype={'protein1': str, 'protein2': str, 'label': int},
            header=None
        )
        
        # 验证标签
        unique_labels = self.ppi_df['label'].unique()
        assert set(unique_labels).issubset({0, 1}), f"标签必须是0或1，发现: {unique_labels}"
        
        # 统计信息
        n_positive = (self.ppi_df['label'] == 1).sum()
        n_negative = (self.ppi_df['label'] == 0).sum()
        logger.info(f"  正样本: {n_positive}, 负样本: {n_negative}, 比例: {n_positive/len(self.ppi_df):.2%}")
    
    def _load_sequences(self):
        """加载蛋白质序列"""
        logger.info(f"加载序列文件: {self.config.fasta_file}")
        
        self.sequences = {}
        with open(self.config.fasta_file) as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq_str = str(record.seq)
                
                # 长度过滤
                if len(seq_str) > self.max_length:
                    logger.warning(f"序列 {record.id} 长度 {len(seq_str)} 超过限制 {self.max_length}，将截断")
                    seq_str = seq_str[:self.max_length]
                
                self.sequences[record.id] = seq_str
        
        # 验证PPI对中的蛋白质是否都有序列
        all_proteins = set(self.ppi_df['protein1']).union(set(self.ppi_df['protein2']))
        missing_proteins = all_proteins - set(self.sequences.keys())
        
        if missing_proteins:
            logger.warning(f"⚠️  {len(missing_proteins)} 个蛋白质缺少序列")
            # 过滤掉缺少序列的PPI对
            self.ppi_df = self.ppi_df[
                self.ppi_df['protein1'].isin(self.sequences) &
                self.ppi_df['protein2'].isin(self.sequences)
            ].reset_index(drop=True)
            logger.info(f"过滤后剩余 {len(self.ppi_df)} 个PPI对")
    
    def __len__(self) -> int:
        return len(self.ppi_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        获取一个样本
        
        Returns:
            Dict包含:
                - seq1: 第一个蛋白质序列（str）
                - seq2: 第二个蛋白质序列（str）
                - label: 标签（int, 0或1）
                - protein1_id: 第一个蛋白质ID（可选）
                - protein2_id: 第二个蛋白质ID（可选）
        """
        row = self.ppi_df.iloc[idx]
        
        protein1_id = row['protein1']
        protein2_id = row['protein2']
        label = row['label']
        
        seq1 = self.sequences[protein1_id]
        seq2 = self.sequences[protein2_id]
        
        # 应用转换
        if self.transform is not None:
            seq1 = self.transform(seq1)
            seq2 = self.transform(seq2)
        
        sample = {
            'seq1': seq1,
            'seq2': seq2,
            'label': label
        }
        
        if self.return_ids:
            sample['protein1_id'] = protein1_id
            sample['protein2_id'] = protein2_id
        
        return sample
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'num_pairs': len(self.ppi_df),
            'num_proteins': len(self.sequences),
            'num_positive': (self.ppi_df['label'] == 1).sum(),
            'num_negative': (self.ppi_df['label'] == 0).sum(),
            'positive_ratio': (self.ppi_df['label'] == 1).sum() / len(self.ppi_df),
            'avg_seq_length': sum(len(s) for s in self.sequences.values()) / len(self.sequences),
            'max_seq_length': max(len(s) for s in self.sequences.values()),
            'min_seq_length': min(len(s) for s in self.sequences.values())
        }
        return stats


class PRINGGraphDataset(Dataset):
    """
    PRING图数据集
    
    用于图重建任务：预测all-against-all的PPI对，然后重建完整网络
    """
    
    def __init__(
        self,
        config: PRINGConfig,
        transform: Optional[Callable] = None,
        max_length: int = 1000,
        load_graph: bool = True
    ):
        """
        初始化图数据集
        
        Args:
            config: PRING配置对象（split应为"all_test"）
            transform: 序列转换函数
            max_length: 序列最大长度
            load_graph: 是否加载真实图（用于评估）
        """
        self.config = config
        self.transform = transform
        self.max_length = max_length
        self.load_graph = load_graph
        
        # 验证配置
        if not config.validate():
            raise FileNotFoundError("配置验证失败，请检查文件路径")
        
        # 加载数据
        logger.info(f"加载PRING图数据: {config}")
        self._load_ppi_pairs()
        self._load_sequences()
        
        if load_graph:
            self._load_ground_truth_graph()
        
        logger.info(f"图数据加载完成: {len(self)} 个候选PPI对")
    
    def _load_ppi_pairs(self):
        """加载all-against-all PPI对"""
        logger.info(f"加载PPI文件: {self.config.ppi_file}")
        
        self.ppi_df = pd.read_csv(
            self.config.ppi_file,
            sep='\t',
            names=['protein1', 'protein2', 'label'],
            dtype={'protein1': str, 'protein2': str, 'label': int},
            header=None
        )
        
        logger.info(f"  总PPI对: {len(self.ppi_df)}")
    
    def _load_sequences(self):
        """加载蛋白质序列"""
        logger.info(f"加载序列文件: {self.config.fasta_file}")
        
        self.sequences = {}
        with open(self.config.fasta_file) as f:
            for record in SeqIO.parse(f, 'fasta'):
                seq_str = str(record.seq)
                if len(seq_str) > self.max_length:
                    seq_str = seq_str[:self.max_length]
                self.sequences[record.id] = seq_str
    
    def _load_ground_truth_graph(self):
        """加载真实图（用于评估）"""
        if self.config.test_graph_file.exists():
            logger.info(f"加载真实图: {self.config.test_graph_file}")
            with open(self.config.test_graph_file, 'rb') as f:
                self.ground_truth_graph = pickle.load(f)
            logger.info(f"  节点数: {self.ground_truth_graph.number_of_nodes()}")
            logger.info(f"  边数: {self.ground_truth_graph.number_of_edges()}")
        else:
            logger.warning(f"真实图文件不存在: {self.config.test_graph_file}")
            self.ground_truth_graph = None
    
    def __len__(self) -> int:
        return len(self.ppi_df)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本（同PRINGPairDataset）"""
        row = self.ppi_df.iloc[idx]
        
        protein1_id = row['protein1']
        protein2_id = row['protein2']
        label = row['label']
        
        seq1 = self.sequences.get(protein1_id, "")
        seq2 = self.sequences.get(protein2_id, "")
        
        if self.transform is not None:
            seq1 = self.transform(seq1)
            seq2 = self.transform(seq2)
        
        return {
            'seq1': seq1,
            'seq2': seq2,
            'label': label,
            'protein1_id': protein1_id,
            'protein2_id': protein2_id
        }
    
    def get_all_proteins(self) -> List[str]:
        """获取所有蛋白质ID"""
        proteins = set(self.ppi_df['protein1']).union(set(self.ppi_df['protein2']))
        return list(proteins)
    
    def save_predictions(self, predictions: List[Tuple[str, str, int]], output_file: Path):
        """
        保存预测结果（用于PRING评估脚本）
        
        Args:
            predictions: List of (protein1_id, protein2_id, predicted_label)
            output_file: 输出文件路径
        """
        with open(output_file, 'w') as f:
            for p1, p2, label in predictions:
                f.write(f"{p1} {p2} {label}\n")
        
        logger.info(f"预测结果已保存到: {output_file}")


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        dataset: PRING数据集
        batch_size: 批大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        **kwargs: 其他DataLoader参数
    
    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )

