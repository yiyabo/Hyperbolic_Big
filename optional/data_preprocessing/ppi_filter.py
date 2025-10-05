#!/usr/bin/env python3
"""
PPI数据过滤器
处理protein.links.detailed数据，支持多重过滤策略
"""

import pandas as pd
import numpy as np
import sqlite3
import gzip
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import networkx as nx

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PPIDataFilter:
    """PPI网络数据过滤器"""
    
    def __init__(self, data_dir: str = "data", confidence_threshold: float = 0.7):
        """
        初始化PPI数据过滤器
        
        Args:
            data_dir: 数据目录路径
            confidence_threshold: 置信度阈值 (0-1)
        """
        self.data_dir = Path(data_dir)
        self.confidence_threshold = confidence_threshold
        self.db_path = self.data_dir / "string_data.db"
        
        # 文件路径
        self.ppi_detailed_file = self.data_dir / "protein.links.detailed.v12.0.txt.gz"
        self.protein_info_file = self.data_dir / "protein.info.v12.0.txt.gz"
        
        # 统计信息
        self.stats = {}
        
    def load_protein_quality_info(self) -> pd.DataFrame:
        """加载蛋白质质量信息用于过滤"""
        logger.info("加载蛋白质质量信息...")
        
        protein_info = []
        chunk_size = 10000
        
        with gzip.open(self.protein_info_file, 'rt') as f:
            # 跳过头部
            header = f.readline().strip().split('\t')
            
            chunk = []
            for line in tqdm(f, desc="读取蛋白质信息"):
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    protein_id = parts[0]
                    protein_name = parts[1]
                    protein_size = int(parts[2])
                    annotation = parts[3] if len(parts) > 3 else ""
                    
                    chunk.append({
                        'protein_id': protein_id,
                        'protein_name': protein_name,
                        'protein_size': protein_size,
                        'annotation': annotation
                    })
                    
                    if len(chunk) >= chunk_size:
                        protein_info.extend(chunk)
                        chunk = []
            
            # 处理剩余数据
            if chunk:
                protein_info.extend(chunk)
        
        df = pd.DataFrame(protein_info)
        logger.info(f"加载了 {len(df)} 个蛋白质的信息")
        return df
    
    def filter_proteins_by_quality(self, protein_df: pd.DataFrame) -> pd.DataFrame:
        """基于质量标准过滤蛋白质"""
        logger.info("基于质量标准过滤蛋白质...")
        
        initial_count = len(protein_df)
        
        # 1. 长度过滤
        length_filtered = protein_df[
            (protein_df['protein_size'] >= 50) &      # 最短50个氨基酸
            (protein_df['protein_size'] <= 5000)      # 最长5000个氨基酸
        ].copy()
        
        # 2. 注释质量过滤
        quality_keywords = [
            'Hypothetical protein',
            'Uncharacterized protein', 
            'Putative',
            'Fragment',
            'Incomplete',
            'too short',
            'missing start',
            'missing stop'
        ]
        
        # 创建质量评分
        length_filtered['quality_score'] = 1.0
        
        for keyword in quality_keywords:
            mask = length_filtered['annotation'].str.contains(keyword, case=False, na=False)
            length_filtered.loc[mask, 'quality_score'] -= 0.2
        
        # 保留质量评分 >= 0.4 的蛋白质
        final_filtered = length_filtered[length_filtered['quality_score'] >= 0.4].copy()
        
        # 统计信息
        self.stats['protein_filtering'] = {
            'initial_count': initial_count,
            'after_length_filter': len(length_filtered),
            'after_quality_filter': len(final_filtered),
            'retention_rate': len(final_filtered) / initial_count
        }
        
        logger.info(f"蛋白质过滤结果:")
        logger.info(f"  初始数量: {initial_count:,}")
        logger.info(f"  长度过滤后: {len(length_filtered):,}")
        logger.info(f"  质量过滤后: {len(final_filtered):,}")
        logger.info(f"  保留率: {len(final_filtered)/initial_count:.1%}")
        
        return final_filtered
    
    def load_and_filter_ppi_data(self, valid_proteins: set) -> pd.DataFrame:
        """加载并过滤PPI数据"""
        logger.info(f"加载PPI数据，置信度阈值: {self.confidence_threshold}")
        
        ppi_data = []
        chunk_size = 50000
        threshold_score = int(self.confidence_threshold * 1000)  # STRING分数是0-1000
        
        with gzip.open(self.ppi_detailed_file, 'rt') as f:
            # 读取头部
            header = f.readline().strip().split()
            logger.info(f"PPI数据列: {header}")
            
            chunk = []
            total_lines = 0
            filtered_lines = 0
            
            for line in tqdm(f, desc="处理PPI数据"):
                parts = line.strip().split()
                if len(parts) >= 10:
                    total_lines += 1
                    
                    protein1 = parts[0]
                    protein2 = parts[1]
                    combined_score = int(parts[9])
                    
                    # 置信度过滤
                    if combined_score >= threshold_score:
                        # 蛋白质质量过滤
                        if protein1 in valid_proteins and protein2 in valid_proteins:
                            filtered_lines += 1
                            
                            chunk.append({
                                'protein1': protein1,
                                'protein2': protein2,
                                'neighborhood': int(parts[2]),
                                'fusion': int(parts[3]),
                                'cooccurence': int(parts[4]),
                                'coexpression': int(parts[5]),
                                'experimental': int(parts[6]),
                                'database': int(parts[7]),
                                'textmining': int(parts[8]),
                                'combined_score': combined_score
                            })
                            
                            if len(chunk) >= chunk_size:
                                ppi_data.extend(chunk)
                                chunk = []
            
            # 处理剩余数据
            if chunk:
                ppi_data.extend(chunk)
        
        df = pd.DataFrame(ppi_data)
        
        # 统计信息
        self.stats['ppi_filtering'] = {
            'total_interactions_raw': total_lines,
            'after_confidence_filter': filtered_lines,
            'after_protein_filter': len(df),
            'confidence_retention_rate': filtered_lines / total_lines if total_lines > 0 else 0,
            'final_retention_rate': len(df) / total_lines if total_lines > 0 else 0
        }
        
        logger.info(f"PPI数据过滤结果:")
        logger.info(f"  原始相互作用: {total_lines:,}")
        logger.info(f"  置信度过滤后: {filtered_lines:,}")
        logger.info(f"  最终保留: {len(df):,}")
        logger.info(f"  最终保留率: {len(df)/total_lines:.1%}")
        
        return df
    
    def analyze_graph_connectivity(self, ppi_df: pd.DataFrame) -> Dict:
        """分析图的连通性"""
        logger.info("分析图连通性...")
        
        # 构建图
        G = nx.Graph()
        
        # 添加边
        edges = [(row['protein1'], row['protein2']) for _, row in ppi_df.iterrows()]
        G.add_edges_from(edges)
        
        # 分析连通分量
        connected_components = list(nx.connected_components(G))
        component_sizes = [len(comp) for comp in connected_components]
        
        # 获取最大连通分量
        largest_component = max(connected_components, key=len)
        
        connectivity_stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'num_components': len(connected_components),
            'largest_component_size': len(largest_component),
            'largest_component_proteins': largest_component,
            'component_size_distribution': {
                'mean': np.mean(component_sizes),
                'std': np.std(component_sizes),
                'min': min(component_sizes),
                'max': max(component_sizes),
                'percentiles': np.percentile(component_sizes, [25, 50, 75, 90, 95, 99])
            }
        }
        
        logger.info(f"图连通性分析结果:")
        logger.info(f"  节点数: {connectivity_stats['total_nodes']:,}")
        logger.info(f"  边数: {connectivity_stats['total_edges']:,}")
        logger.info(f"  连通分量数: {connectivity_stats['num_components']:,}")
        logger.info(f"  最大连通分量大小: {connectivity_stats['largest_component_size']:,}")
        logger.info(f"  最大分量占比: {connectivity_stats['largest_component_size']/connectivity_stats['total_nodes']:.1%}")
        
        self.stats['connectivity'] = connectivity_stats
        return connectivity_stats
    
    def filter_to_largest_component(self, ppi_df: pd.DataFrame, largest_component: set) -> pd.DataFrame:
        """过滤到最大连通分量"""
        logger.info("过滤到最大连通分量...")
        
        filtered_df = ppi_df[
            ppi_df['protein1'].isin(largest_component) & 
            ppi_df['protein2'].isin(largest_component)
        ].copy()
        
        logger.info(f"连通分量过滤结果:")
        logger.info(f"  过滤前相互作用: {len(ppi_df):,}")
        logger.info(f"  过滤后相互作用: {len(filtered_df):,}")
        logger.info(f"  保留率: {len(filtered_df)/len(ppi_df):.1%}")
        
        return filtered_df
    
    def save_filtered_data(self, protein_df: pd.DataFrame, ppi_df: pd.DataFrame, 
                          output_dir: Optional[str] = None) -> Dict[str, Path]:
        """保存过滤后的数据"""
        if output_dir is None:
            output_dir = self.data_dir / "filtered"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        # 保存文件路径
        output_files = {}
        
        # 保存蛋白质信息
        protein_file = output_dir / f"filtered_proteins_conf{self.confidence_threshold}.csv"
        protein_df.to_csv(protein_file, index=False)
        output_files['proteins'] = protein_file
        
        # 保存PPI数据
        ppi_file = output_dir / f"filtered_ppi_conf{self.confidence_threshold}.csv"
        ppi_df.to_csv(ppi_file, index=False)
        output_files['ppi'] = ppi_file
        
        # 保存统计信息
        stats_file = output_dir / f"filtering_stats_conf{self.confidence_threshold}.json"
        import json
        with open(stats_file, 'w') as f:
            # 转换set为list用于JSON序列化
            stats_copy = self.stats.copy()
            if 'connectivity' in stats_copy and 'largest_component_proteins' in stats_copy['connectivity']:
                stats_copy['connectivity']['largest_component_proteins'] = list(
                    stats_copy['connectivity']['largest_component_proteins']
                )
            json.dump(stats_copy, f, indent=2, default=str)
        output_files['stats'] = stats_file
        
        logger.info(f"过滤后的数据已保存到:")
        for name, path in output_files.items():
            logger.info(f"  {name}: {path}")
        
        return output_files
    
    def run_complete_filtering(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """运行完整的数据过滤流程"""
        logger.info(f"开始完整的数据过滤流程，置信度阈值: {self.confidence_threshold}")
        
        # 1. 加载蛋白质信息
        protein_df = self.load_protein_quality_info()
        
        # 2. 过滤蛋白质质量
        filtered_proteins = self.filter_proteins_by_quality(protein_df)
        valid_protein_set = set(filtered_proteins['protein_id'])
        
        # 3. 加载并过滤PPI数据
        filtered_ppi = self.load_and_filter_ppi_data(valid_protein_set)
        
        # 4. 分析图连通性
        connectivity_stats = self.analyze_graph_connectivity(filtered_ppi)
        
        # 5. 过滤到最大连通分量
        final_ppi = self.filter_to_largest_component(
            filtered_ppi, 
            connectivity_stats['largest_component_proteins']
        )
        
        # 6. 更新蛋白质列表(只保留在最大连通分量中的)
        proteins_in_graph = set(final_ppi['protein1']).union(set(final_ppi['protein2']))
        final_proteins = filtered_proteins[
            filtered_proteins['protein_id'].isin(proteins_in_graph)
        ].copy()
        
        # 7. 保存结果
        output_files = self.save_filtered_data(final_proteins, final_ppi)
        
        # 8. 最终统计
        logger.info("=" * 60)
        logger.info("最终数据过滤结果总结:")
        logger.info(f"  最终蛋白质数量: {len(final_proteins):,}")
        logger.info(f"  最终相互作用数量: {len(final_ppi):,}")
        logger.info(f"  平均度数: {2*len(final_ppi)/len(final_proteins):.2f}")
        logger.info(f"  置信度阈值: {self.confidence_threshold}")
        logger.info("=" * 60)
        
        return final_proteins, final_ppi, output_files

def main():
    """主函数 - 命令行使用"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PPI数据过滤器')
    parser.add_argument('--data-dir', default='data', help='数据目录路径')
    parser.add_argument('--confidence', type=float, default=0.7, help='置信度阈值(0-1)')
    parser.add_argument('--output-dir', help='输出目录路径')
    
    args = parser.parse_args()
    
    # 创建过滤器并运行
    filter_obj = PPIDataFilter(
        data_dir=args.data_dir,
        confidence_threshold=args.confidence
    )
    
    proteins, ppi, output_files = filter_obj.run_complete_filtering()
    
    print(f"\n过滤完成！输出文件:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()