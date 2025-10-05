"""
PRING数据加载器模块

提供标准化的PyTorch Dataset类用于加载PRING基准测试数据
"""

from .pring_dataset import (
    PRINGPairDataset,
    PRINGGraphDataset,
    get_dataloader
)
from .config import PRINGConfig

__all__ = [
    'PRINGPairDataset',
    'PRINGGraphDataset',
    'PRINGConfig',
    'get_dataloader'
]

__version__ = '1.0.0'

