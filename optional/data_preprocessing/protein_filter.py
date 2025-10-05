#!/usr/bin/env python3
"""
蛋白质质量过滤器
基于序列长度、注释质量等标准过滤蛋白质
"""

import pandas as pd
import numpy as np
import gzip
from pathlib import Path
import logging
from typing import Dict, List, Set
import re

logger = logging.getLogger(__name__)

class ProteinQualityFilter:
    """蛋白质质量过滤器"""
    
    def __init__(self):
        """初始化过滤器"""
        
        # 质量控制参数
        self.length_limits = {
            'min_length': 50,      # 最短50个氨基酸
            'max_length': 5000,    # 最长5000个氨基酸
            'optimal_min': 100,    # 最佳最小长度
            'optimal_max': 1000    # 最佳最大长度
        }
        
        # 低质量关键词(降低质量分数)
        self.low_quality_keywords = [
            'hypothetical protein',
            'uncharacterized protein',
            'putative',
            'fragment',
            'incomplete', 
            'too short',
            'missing start',
            'missing stop',
            'partial',
            'truncated'
        ]
        
        # 高质量关键词(提高质量分数)
        self.high_quality_keywords = [
            'characterized',
            'crystal structure',
            'experimentally verified',
            'well-studied',
            'enzyme',
            'kinase',
            'ligase',
            'reductase',
            'transferase',
            'hydrolase'
        ]
        
        # 预测方法关键词(中等质量)
        self.prediction_keywords = [
            'derived by automated computational analysis',
            'gene prediction method',
            'protein homology',
            'genemark',
            'similarity'
        ]
    
    def calculate_length_score(self, length: int) -> float:
        """基于序列长度计算质量分数"""
        if length < self.length_limits['min_length'] or length > self.length_limits['max_length']:
            return 0.0
        
        if self.length_limits['optimal_min'] <= length <= self.length_limits['optimal_max']:
            return 1.0
        
        # 对于过短或过长的序列给予部分分数
        if length < self.length_limits['optimal_min']:
            ratio = (length - self.length_limits['min_length']) / \
                    (self.length_limits['optimal_min'] - self.length_limits['min_length'])
            return 0.5 + 0.5 * ratio
        else:  # length > optimal_max
            ratio = (self.length_limits['max_length'] - length) / \
                    (self.length_limits['max_length'] - self.length_limits['optimal_max'])
            return 0.5 + 0.5 * ratio
    
    def calculate_annotation_score(self, annotation: str) -> float:
        """基于注释质量计算分数"""
        if pd.isna(annotation) or annotation.strip() == '':
            return 0.1  # 无注释给最低分
        
        annotation_lower = annotation.lower()
        score = 0.5  # 基础分数
        
        # 检查低质量关键词
        for keyword in self.low_quality_keywords:
            if keyword in annotation_lower:
                score -= 0.15
        
        # 检查高质量关键词
        for keyword in self.high_quality_keywords:
            if keyword in annotation_lower:
                score += 0.2
        
        # 检查预测方法关键词
        for keyword in self.prediction_keywords:
            if keyword in annotation_lower:
                score -= 0.05
        
        # 注释长度奖励(详细的注释通常质量更高)
        if len(annotation) > 100:
            score += 0.1
        elif len(annotation) > 200:
            score += 0.15
        
        # 限制分数范围
        return max(0.0, min(1.0, score))
    
    def calculate_name_score(self, protein_name: str) -> float:
        """基于蛋白质名称计算分数"""
        if pd.isna(protein_name) or protein_name.strip() == '':
            return 0.1
        
        name_lower = protein_name.lower()
        score = 0.5
        
        # 有意义的基因名通常质量更高
        if re.match(r'^[a-z]{2,5}\d*[a-z]*$', name_lower):  # 如: acsA, hsp70, etc.
            score += 0.2
        
        # 通用ID格式通常质量较低
        if re.match(r'^[a-z]+\d+\.\d+$', name_lower):  # 如: ABC123.1
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def extract_species_id(self, protein_id: str) -> int:
        """从蛋白质ID中提取物种ID"""
        try:
            return int(protein_id.split('.')[0])
        except:
            return -1
    
    def calculate_species_score(self, species_id: int) -> float:
        """基于物种计算质量分数"""
        # 模式生物给予更高分数
        model_organisms = {
            9606: 1.0,    # Homo sapiens
            10090: 1.0,   # Mus musculus  
            7227: 0.9,    # Drosophila melanogaster
            6239: 0.9,    # Caenorhabditis elegans
            3702: 0.9,    # Arabidopsis thaliana
            4932: 0.9,    # Saccharomyces cerevisiae
            511145: 0.8,  # Escherichia coli
            83333: 0.8,   # Escherichia coli K-12
        }
        
        return model_organisms.get(species_id, 0.6)  # 其他物种默认0.6分
    
    def calculate_overall_quality_score(self, protein_data: Dict) -> float:
        """计算综合质量分数"""
        
        # 各项分数权重
        weights = {
            'length': 0.3,
            'annotation': 0.4,
            'name': 0.2,
            'species': 0.1
        }
        
        # 计算各项分数
        length_score = self.calculate_length_score(protein_data['protein_size'])
        annotation_score = self.calculate_annotation_score(protein_data['annotation'])
        name_score = self.calculate_name_score(protein_data['protein_name'])
        species_score = self.calculate_species_score(protein_data['species_id'])
        
        # 加权平均
        overall_score = (
            weights['length'] * length_score +
            weights['annotation'] * annotation_score +
            weights['name'] * name_score +
            weights['species'] * species_score
        )
        
        return overall_score
    
    def filter_proteins(self, protein_df: pd.DataFrame, 
                       quality_threshold: float = 0.5) -> pd.DataFrame:
        """过滤蛋白质并添加质量分数"""
        
        logger.info(f"开始蛋白质质量过滤，质量阈值: {quality_threshold}")
        
        # 添加物种ID列
        protein_df['species_id'] = protein_df['protein_id'].apply(self.extract_species_id)
        
        # 计算各项质量分数
        protein_df['length_score'] = protein_df['protein_size'].apply(self.calculate_length_score)
        protein_df['annotation_score'] = protein_df['annotation'].apply(self.calculate_annotation_score)
        protein_df['name_score'] = protein_df['protein_name'].apply(self.calculate_name_score)
        protein_df['species_score'] = protein_df['species_id'].apply(self.calculate_species_score)
        
        # 计算综合质量分数
        protein_df['quality_score'] = protein_df.apply(
            lambda row: self.calculate_overall_quality_score(row.to_dict()), 
            axis=1
        )
        
        # 应用质量阈值过滤
        filtered_df = protein_df[protein_df['quality_score'] >= quality_threshold].copy()
        
        # 统计信息
        stats = {
            'initial_count': len(protein_df),
            'filtered_count': len(filtered_df),
            'retention_rate': len(filtered_df) / len(protein_df),
            'quality_distribution': {
                'mean': protein_df['quality_score'].mean(),
                'std': protein_df['quality_score'].std(),
                'min': protein_df['quality_score'].min(),
                'max': protein_df['quality_score'].max(),
                'percentiles': protein_df['quality_score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
            },
            'filtered_quality_distribution': {
                'mean': filtered_df['quality_score'].mean(),
                'std': filtered_df['quality_score'].std(), 
                'min': filtered_df['quality_score'].min(),
                'max': filtered_df['quality_score'].max()
            }
        }
        
        logger.info(f"蛋白质质量过滤结果:")
        logger.info(f"  初始数量: {stats['initial_count']:,}")
        logger.info(f"  过滤后数量: {stats['filtered_count']:,}")
        logger.info(f"  保留率: {stats['retention_rate']:.1%}")
        logger.info(f"  平均质量分数: {stats['quality_distribution']['mean']:.3f}")
        logger.info(f"  过滤后平均质量分数: {stats['filtered_quality_distribution']['mean']:.3f}")
        
        return filtered_df, stats
    
    def get_quality_categories(self, protein_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """将蛋白质按质量分类"""
        
        categories = {
            'high_quality': protein_df[protein_df['quality_score'] >= 0.8],
            'medium_quality': protein_df[
                (protein_df['quality_score'] >= 0.6) & 
                (protein_df['quality_score'] < 0.8)
            ],
            'low_quality': protein_df[
                (protein_df['quality_score'] >= 0.4) & 
                (protein_df['quality_score'] < 0.6)
            ],
            'very_low_quality': protein_df[protein_df['quality_score'] < 0.4]
        }
        
        logger.info("蛋白质质量分类:")
        for category, df in categories.items():
            logger.info(f"  {category}: {len(df):,} ({len(df)/len(protein_df):.1%})")
        
        return categories
    
    def analyze_quality_by_species(self, protein_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """按物种分析质量分布"""
        
        species_stats = protein_df.groupby('species_id').agg({
            'quality_score': ['count', 'mean', 'std', 'min', 'max'],
            'protein_size': 'mean'
        }).round(3)
        
        species_stats.columns = ['count', 'quality_mean', 'quality_std', 'quality_min', 'quality_max', 'avg_length']
        species_stats = species_stats.sort_values('count', ascending=False).head(top_n)
        
        logger.info(f"前{top_n}个物种的质量统计:")
        for species_id, row in species_stats.iterrows():
            logger.info(f"  物种{species_id}: {row['count']:,}个蛋白质, 平均质量{row['quality_mean']:.3f}")
        
        return species_stats