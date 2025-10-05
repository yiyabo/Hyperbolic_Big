#!/usr/bin/env python3
"""
数据统计分析器
提供全面的数据统计和可视化功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

logger = logging.getLogger(__name__)

class DataStatistics:
    """数据统计分析器"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        """初始化统计分析器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_ppi_network_properties(self, ppi_df: pd.DataFrame) -> Dict:
        """分析PPI网络的基本属性"""
        logger.info("分析PPI网络基本属性...")
        
        # 基础统计
        stats = {
            'num_interactions': len(ppi_df),
            'num_proteins': len(set(ppi_df['protein1']).union(set(ppi_df['protein2']))),
            'avg_degree': 2 * len(ppi_df) / len(set(ppi_df['protein1']).union(set(ppi_df['protein2']))),
        }
        
        # 置信度分析
        stats['confidence_stats'] = {
            'mean_combined_score': ppi_df['combined_score'].mean(),
            'std_combined_score': ppi_df['combined_score'].std(),
            'min_combined_score': ppi_df['combined_score'].min(),
            'max_combined_score': ppi_df['combined_score'].max(),
            'percentiles': ppi_df['combined_score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        }
        
        # 证据通道分析
        evidence_channels = ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 
                           'experimental', 'database', 'textmining']
        
        stats['evidence_analysis'] = {}
        for channel in evidence_channels:
            if channel in ppi_df.columns:
                stats['evidence_analysis'][channel] = {
                    'non_zero_count': (ppi_df[channel] > 0).sum(),
                    'non_zero_percentage': (ppi_df[channel] > 0).mean() * 100,
                    'mean_score': ppi_df[channel].mean(),
                    'max_score': ppi_df[channel].max()
                }
        
        logger.info(f"PPI网络基本属性:")
        logger.info(f"  相互作用数: {stats['num_interactions']:,}")
        logger.info(f"  蛋白质数: {stats['num_proteins']:,}")
        logger.info(f"  平均度数: {stats['avg_degree']:.2f}")
        logger.info(f"  平均置信度: {stats['confidence_stats']['mean_combined_score']:.1f}")
        
        return stats
    
    def analyze_protein_properties(self, protein_df: pd.DataFrame) -> Dict:
        """分析蛋白质属性分布"""
        logger.info("分析蛋白质属性分布...")
        
        stats = {
            'num_proteins': len(protein_df),
            'length_distribution': {
                'mean': protein_df['protein_size'].mean(),
                'std': protein_df['protein_size'].std(),
                'min': protein_df['protein_size'].min(),
                'max': protein_df['protein_size'].max(),
                'percentiles': protein_df['protein_size'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
            }
        }
        
        # 物种分布分析
        if 'species_id' in protein_df.columns:
            species_counts = protein_df['species_id'].value_counts()
            stats['species_distribution'] = {
                'num_species': len(species_counts),
                'top_10_species': species_counts.head(10).to_dict(),
                'species_with_single_protein': (species_counts == 1).sum(),
                'species_with_100plus_proteins': (species_counts >= 100).sum()
            }
        
        # 质量分数分析
        if 'quality_score' in protein_df.columns:
            stats['quality_distribution'] = {
                'mean': protein_df['quality_score'].mean(),
                'std': protein_df['quality_score'].std(),
                'min': protein_df['quality_score'].min(),
                'max': protein_df['quality_score'].max(),
                'high_quality_count': (protein_df['quality_score'] >= 0.8).sum(),
                'medium_quality_count': ((protein_df['quality_score'] >= 0.6) & 
                                       (protein_df['quality_score'] < 0.8)).sum(),
                'low_quality_count': (protein_df['quality_score'] < 0.6).sum()
            }
        
        logger.info(f"蛋白质属性分析:")
        logger.info(f"  蛋白质总数: {stats['num_proteins']:,}")
        logger.info(f"  平均长度: {stats['length_distribution']['mean']:.1f}")
        logger.info(f"  长度范围: {stats['length_distribution']['min']}-{stats['length_distribution']['max']}")
        
        return stats
    
    def plot_ppi_statistics(self, ppi_df: pd.DataFrame, save_plots: bool = True) -> None:
        """绘制PPI网络统计图表"""
        logger.info("绘制PPI网络统计图表...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPI Network Statistics', fontsize=16)
        
        # 1. 置信度分布
        axes[0,0].hist(ppi_df['combined_score'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Combined Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Confidence Score Distribution')
        axes[0,0].axvline(ppi_df['combined_score'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {ppi_df["combined_score"].mean():.1f}')
        axes[0,0].legend()
        
        # 2. 证据通道使用情况
        evidence_channels = ['neighborhood', 'fusion', 'cooccurence', 'coexpression', 
                           'experimental', 'database', 'textmining']
        available_channels = [ch for ch in evidence_channels if ch in ppi_df.columns]
        
        channel_usage = []
        channel_names = []
        for channel in available_channels:
            usage = (ppi_df[channel] > 0).mean() * 100
            channel_usage.append(usage)
            channel_names.append(channel.capitalize())
        
        axes[0,1].bar(channel_names, channel_usage)
        axes[0,1].set_ylabel('Usage Percentage (%)')
        axes[0,1].set_title('Evidence Channel Usage')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. 度数分布
        all_proteins = list(ppi_df['protein1']) + list(ppi_df['protein2'])
        degree_counts = pd.Series(all_proteins).value_counts()
        
        axes[0,2].hist(degree_counts.values, bins=50, alpha=0.7, edgecolor='black')
        axes[0,2].set_xlabel('Degree')
        axes[0,2].set_ylabel('Number of Proteins')
        axes[0,2].set_title('Degree Distribution')
        axes[0,2].set_yscale('log')
        axes[0,2].set_xscale('log')
        
        # 4. 证据通道相关性热图
        if len(available_channels) > 1:
            corr_matrix = ppi_df[available_channels].corr()
            im = axes[1,0].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1,0].set_xticks(range(len(available_channels)))
            axes[1,0].set_yticks(range(len(available_channels)))
            axes[1,0].set_xticklabels([ch.capitalize() for ch in available_channels], rotation=45)
            axes[1,0].set_yticklabels([ch.capitalize() for ch in available_channels])
            axes[1,0].set_title('Evidence Channel Correlation')
            
            # 添加相关系数标注
            for i in range(len(available_channels)):
                for j in range(len(available_channels)):
                    axes[1,0].text(j, i, f'{corr_matrix.iloc[i,j]:.2f}', 
                                  ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=axes[1,0])
        
        # 5. 置信度vs证据通道散点图
        if 'experimental' in ppi_df.columns:
            axes[1,1].scatter(ppi_df['experimental'], ppi_df['combined_score'], 
                            alpha=0.5, s=1)
            axes[1,1].set_xlabel('Experimental Evidence')
            axes[1,1].set_ylabel('Combined Score')
            axes[1,1].set_title('Experimental vs Combined Score')
        
        # 6. 累积分布函数
        sorted_scores = np.sort(ppi_df['combined_score'])
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1,2].plot(sorted_scores, cumulative)
        axes[1,2].set_xlabel('Combined Score')
        axes[1,2].set_ylabel('Cumulative Probability')
        axes[1,2].set_title('Cumulative Distribution Function')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "ppi_network_statistics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"PPI统计图表已保存: {plot_path}")
        
        plt.show()
    
    def plot_protein_statistics(self, protein_df: pd.DataFrame, save_plots: bool = True) -> None:
        """绘制蛋白质统计图表"""
        logger.info("绘制蛋白质统计图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Protein Statistics', fontsize=16)
        
        # 1. 长度分布
        axes[0,0].hist(protein_df['protein_size'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Protein Length (amino acids)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Protein Length Distribution')
        axes[0,0].axvline(protein_df['protein_size'].mean(), color='red', linestyle='--',
                         label=f'Mean: {protein_df["protein_size"].mean():.1f}')
        axes[0,0].legend()
        
        # 2. 物种分布
        if 'species_id' in protein_df.columns:
            species_counts = protein_df['species_id'].value_counts().head(15)
            axes[0,1].bar(range(len(species_counts)), species_counts.values)
            axes[0,1].set_xlabel('Species (Top 15)')
            axes[0,1].set_ylabel('Number of Proteins')
            axes[0,1].set_title('Protein Distribution by Species')
            axes[0,1].set_xticks(range(len(species_counts)))
            axes[0,1].set_xticklabels(species_counts.index, rotation=45)
        
        # 3. 质量分数分布
        if 'quality_score' in protein_df.columns:
            axes[1,0].hist(protein_df['quality_score'], bins=30, alpha=0.7, edgecolor='black')
            axes[1,0].set_xlabel('Quality Score')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Quality Score Distribution')
            axes[1,0].axvline(protein_df['quality_score'].mean(), color='red', linestyle='--',
                             label=f'Mean: {protein_df["quality_score"].mean():.3f}')
            axes[1,0].legend()
        
        # 4. 长度vs质量分数散点图
        if 'quality_score' in protein_df.columns:
            axes[1,1].scatter(protein_df['protein_size'], protein_df['quality_score'], 
                            alpha=0.5, s=1)
            axes[1,1].set_xlabel('Protein Length')
            axes[1,1].set_ylabel('Quality Score')
            axes[1,1].set_title('Length vs Quality Score')
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / "protein_statistics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"蛋白质统计图表已保存: {plot_path}")
        
        plt.show()
    
    def generate_comprehensive_report(self, protein_df: pd.DataFrame, 
                                    ppi_df: pd.DataFrame, 
                                    expert_groups: Dict[str, List[str]] = None,
                                    filtering_stats: Dict = None) -> Dict:
        """生成综合数据报告"""
        logger.info("生成综合数据报告...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'proteins': len(protein_df),
                'interactions': len(ppi_df),
                'unique_proteins_in_network': len(set(ppi_df['protein1']).union(set(ppi_df['protein2'])))
            }
        }
        
        # 添加各种统计信息
        report['protein_analysis'] = self.analyze_protein_properties(protein_df)
        report['ppi_analysis'] = self.analyze_ppi_network_properties(ppi_df)
        
        # 专家组分析
        if expert_groups:
            report['expert_groups_analysis'] = {
                'num_experts': len(expert_groups),
                'total_proteins_in_groups': sum(len(proteins) for proteins in expert_groups.values()),
                'group_sizes': {name: len(proteins) for name, proteins in expert_groups.items()},
                'size_distribution': {
                    'min_size': min(len(proteins) for proteins in expert_groups.values()),
                    'max_size': max(len(proteins) for proteins in expert_groups.values()),
                    'mean_size': np.mean([len(proteins) for proteins in expert_groups.values()]),
                    'std_size': np.std([len(proteins) for proteins in expert_groups.values()])
                }
            }
        
        # 过滤统计信息
        if filtering_stats:
            report['filtering_statistics'] = filtering_stats
        
        # 保存报告
        report_path = self.output_dir / "comprehensive_data_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"综合数据报告已保存: {report_path}")
        
        # 打印关键统计信息
        self._print_summary_report(report)
        
        return report
    
    def _print_summary_report(self, report: Dict) -> None:
        """打印摘要报告"""
        print("\n" + "="*80)
        print("                      数据处理摘要报告")
        print("="*80)
        
        # 基础数据统计
        print(f"\n📊 基础数据统计:")
        print(f"   蛋白质总数: {report['data_summary']['proteins']:,}")
        print(f"   相互作用总数: {report['data_summary']['interactions']:,}")
        print(f"   网络中蛋白质数: {report['data_summary']['unique_proteins_in_network']:,}")
        print(f"   平均度数: {2*report['data_summary']['interactions']/report['data_summary']['unique_proteins_in_network']:.2f}")
        
        # 蛋白质属性
        if 'protein_analysis' in report:
            pa = report['protein_analysis']
            print(f"\n🧬 蛋白质属性:")
            print(f"   平均长度: {pa['length_distribution']['mean']:.1f} ± {pa['length_distribution']['std']:.1f}")
            print(f"   长度范围: {pa['length_distribution']['min']} - {pa['length_distribution']['max']}")
            
            if 'species_distribution' in pa:
                print(f"   物种数量: {pa['species_distribution']['num_species']:,}")
            
            if 'quality_distribution' in pa:
                qd = pa['quality_distribution']
                print(f"   平均质量分数: {qd['mean']:.3f}")
                print(f"   高质量蛋白质: {qd['high_quality_count']:,} ({qd['high_quality_count']/report['data_summary']['proteins']*100:.1f}%)")
        
        # PPI网络属性
        if 'ppi_analysis' in report:
            ppi = report['ppi_analysis']
            print(f"\n🔗 PPI网络属性:")
            print(f"   平均置信度: {ppi['confidence_stats']['mean_combined_score']:.1f}")
            print(f"   置信度范围: {ppi['confidence_stats']['min_combined_score']} - {ppi['confidence_stats']['max_combined_score']}")
            
            if 'evidence_analysis' in ppi:
                print(f"   证据通道使用率:")
                for channel, stats in ppi['evidence_analysis'].items():
                    print(f"     {channel}: {stats['non_zero_percentage']:.1f}%")
        
        # 专家组信息
        if 'expert_groups_analysis' in report:
            ega = report['expert_groups_analysis']
            print(f"\n🎭 MoE专家组:")
            print(f"   专家组数量: {ega['num_experts']}")
            print(f"   覆盖蛋白质: {ega['total_proteins_in_groups']:,}")
            print(f"   平均组大小: {ega['size_distribution']['mean_size']:.1f} ± {ega['size_distribution']['std_size']:.1f}")
            print(f"   组大小范围: {ega['size_distribution']['min_size']} - {ega['size_distribution']['max_size']}")
        
        # 过滤统计
        if 'filtering_statistics' in report:
            print(f"\n📈 数据过滤效果:")
            fs = report['filtering_statistics']
            if 'protein_filtering' in fs:
                pf = fs['protein_filtering']
                print(f"   蛋白质保留率: {pf['retention_rate']*100:.1f}%")
            if 'ppi_filtering' in fs:
                ppif = fs['ppi_filtering']
                print(f"   PPI保留率: {ppif['final_retention_rate']*100:.1f}%")
        
        print("\n" + "="*80)
        print("✅ 数据预处理完成，可以开始PSSM生成！")
        print("="*80)
    
    def save_summary_statistics(self, stats: Dict, filename: str = "summary_stats.json") -> Path:
        """保存摘要统计信息"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"摘要统计信息已保存: {output_path}")
        return output_path