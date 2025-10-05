"""
PRING数据集配置

支持本地和服务器环境的灵活配置
"""

import os
from pathlib import Path
from typing import Optional


class PRINGConfig:
    """PRING数据集配置类"""
    
    # 默认数据根目录（可通过环境变量覆盖）
    DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "PRING" / "data_process" / "pring_dataset"
    
    def __init__(
        self,
        data_root: Optional[str] = None,
        species: str = "human",
        sampling_strategy: str = "BFS",
        split: str = "train"
    ):
        """
        初始化PRING配置
        
        Args:
            data_root: 数据根目录路径。如果为None，自动检测：
                      1. 环境变量 PRING_DATA_ROOT
                      2. 默认路径（项目data/PRING/...）
            species: 物种名称，可选 "human", "arath", "yeast", "ecoli"
            sampling_strategy: 采样策略，可选 "BFS", "DFS", "RANDOM_WALK"
                             （仅对human有效）
            split: 数据切分，可选 "train", "val", "test", "all_test"
        """
        self.species = species
        self.sampling_strategy = sampling_strategy
        self.split = split
        
        # 自动检测数据根目录
        if data_root is None:
            data_root = os.environ.get('PRING_DATA_ROOT', self.DEFAULT_DATA_ROOT)
        
        self.data_root = Path(data_root)
        
        # 验证路径
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"PRING数据根目录不存在: {self.data_root}\n"
                f"请设置环境变量 PRING_DATA_ROOT 或将数据放在默认位置。"
            )
        
        # 构建具体路径
        self._build_paths()
    
    def _build_paths(self):
        """构建具体文件路径"""
        self.species_dir = self.data_root / self.species
        
        if not self.species_dir.exists():
            raise FileNotFoundError(
                f"物种目录不存在: {self.species_dir}\n"
                f"可用物种: human, arath, yeast, ecoli"
            )
        
        # human有多种采样策略
        if self.species == "human":
            self.strategy_dir = self.species_dir / self.sampling_strategy
            
            if not self.strategy_dir.exists():
                raise FileNotFoundError(
                    f"采样策略目录不存在: {self.strategy_dir}\n"
                    f"可用策略: BFS, DFS, RANDOM_WALK"
                )
            
            # PPI文件路径
            if self.split == "train":
                self.ppi_file = self.strategy_dir / "human_train_ppi.txt"
            elif self.split == "val":
                self.ppi_file = self.strategy_dir / "human_val_ppi.txt"
            elif self.split == "test":
                self.ppi_file = self.strategy_dir / "human_test_ppi.txt"
            elif self.split == "all_test":
                self.ppi_file = self.strategy_dir / "all_test_ppi.txt"
            else:
                raise ValueError(f"未知的split: {self.split}")
            
            # 图文件路径
            self.train_graph_file = self.strategy_dir / "human_train_graph.pkl"
            self.test_graph_file = self.strategy_dir / "human_test_graph.pkl"
            self.sampled_nodes_file = self.strategy_dir / "test_sampled_nodes.pkl"
            
        else:
            # 其他物种没有采样策略子目录
            self.strategy_dir = self.species_dir
            
            if self.split == "test":
                self.ppi_file = self.species_dir / f"{self.species}_test_ppi.txt"
            elif self.split == "all_test":
                self.ppi_file = self.species_dir / f"{self.species}_all_test_ppi.txt"
            else:
                raise ValueError(
                    f"物种 {self.species} 只支持 test 和 all_test split"
                )
            
            # 图文件路径
            self.test_graph_file = self.species_dir / f"{self.species}_test_graph.pkl"
            self.sampled_nodes_file = self.species_dir / f"{self.species}_{self.sampling_strategy}_sampled_nodes.pkl"
        
        # 序列文件（所有物种共用）
        self.fasta_file = self.species_dir / f"{self.species}_simple.fasta"
        self.full_fasta_file = self.species_dir / f"{self.species}.fasta"
        
        # 蛋白质ID映射文件
        self.protein_id_file = self.species_dir / f"{self.species}_protein_id.csv"
        
        # 完整PPI和图文件
        self.all_ppi_file = self.species_dir / f"{self.species}_ppi.txt"
        self.full_graph_file = self.species_dir / f"{self.species}_graph.pkl"
    
    def validate(self) -> bool:
        """验证所有必需文件是否存在"""
        required_files = [self.ppi_file, self.fasta_file]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            print(f"⚠️  缺失文件:")
            for f in missing_files:
                print(f"  - {f}")
            return False
        
        return True
    
    def __repr__(self):
        return (
            f"PRINGConfig(\n"
            f"  species={self.species},\n"
            f"  sampling_strategy={self.sampling_strategy},\n"
            f"  split={self.split},\n"
            f"  ppi_file={self.ppi_file},\n"
            f"  fasta_file={self.fasta_file}\n"
            f")"
        )


# 预定义配置
HUMAN_TRAIN_BFS = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
HUMAN_VAL_BFS = PRINGConfig(species="human", sampling_strategy="BFS", split="val")
HUMAN_TEST_BFS = PRINGConfig(species="human", sampling_strategy="BFS", split="test")

ARATH_TEST = PRINGConfig(species="arath", split="test")
YEAST_TEST = PRINGConfig(species="yeast", split="test")
ECOLI_TEST = PRINGConfig(species="ecoli", split="test")

