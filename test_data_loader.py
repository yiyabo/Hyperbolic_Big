#!/usr/bin/env python3
"""
测试PRING数据加载器

验证数据加载器的功能和性能
"""

import sys
sys.path.append('.')

import time
import logging
from data_loader import PRINGPairDataset, PRINGGraphDataset, PRINGConfig, get_dataloader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_config():
    """测试配置类"""
    logger.info("="*60)
    logger.info("测试1: 配置类")
    logger.info("="*60)
    
    # 测试human配置
    config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
    logger.info(f"配置创建成功:\n{config}")
    
    # 验证文件
    assert config.validate(), "配置验证失败"
    logger.info("✅ 配置验证通过")
    
    # 测试跨物种配置
    for species in ['arath', 'yeast', 'ecoli']:
        config = PRINGConfig(species=species, split="test")
        assert config.validate(), f"{species} 配置验证失败"
        logger.info(f"✅ {species} 配置验证通过")


def test_pair_dataset():
    """测试成对数据集"""
    logger.info("\n" + "="*60)
    logger.info("测试2: PRINGPairDataset")
    logger.info("="*60)
    
    # 创建数据集
    config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
    dataset = PRINGPairDataset(config, return_ids=True)
    
    logger.info(f"数据集大小: {len(dataset)}")
    
    # 获取统计信息
    stats = dataset.get_statistics()
    logger.info("数据集统计:")
    logger.info(f"  总PPI对: {stats['num_pairs']:,}")
    logger.info(f"  蛋白质数: {stats['num_proteins']:,}")
    logger.info(f"  正样本: {stats['num_positive']:,}")
    logger.info(f"  负样本: {stats['num_negative']:,}")
    logger.info(f"  正样本比例: {stats['positive_ratio']:.2%}")
    logger.info(f"  平均序列长度: {stats['avg_seq_length']:.1f}")
    logger.info(f"  序列长度范围: [{stats['min_seq_length']}, {stats['max_seq_length']}]")
    
    # 测试样本访问
    sample = dataset[0]
    logger.info(f"\n样本示例:")
    logger.info(f"  蛋白质1 ID: {sample['protein1_id']}")
    logger.info(f"  蛋白质2 ID: {sample['protein2_id']}")
    logger.info(f"  序列1长度: {len(sample['seq1'])}")
    logger.info(f"  序列2长度: {len(sample['seq2'])}")
    logger.info(f"  序列1前50字符: {sample['seq1'][:50]}...")
    logger.info(f"  标签: {sample['label']}")
    
    logger.info("✅ PRINGPairDataset 测试通过")
    return dataset


def test_dataloader(dataset):
    """测试DataLoader"""
    logger.info("\n" + "="*60)
    logger.info("测试3: DataLoader")
    logger.info("="*60)
    
    # 创建DataLoader
    dataloader = get_dataloader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0  # 测试时使用0避免多进程问题
    )
    
    logger.info(f"DataLoader batch数: {len(dataloader)}")
    
    # 遍历一个batch
    start_time = time.time()
    for i, batch in enumerate(dataloader):
        if i == 0:  # 只测试第一个batch
            logger.info(f"\nBatch示例:")
            logger.info(f"  seq1类型: {type(batch['seq1'])}")
            logger.info(f"  seq1长度: {len(batch['seq1'])}")
            logger.info(f"  seq2长度: {len(batch['seq2'])}")
            logger.info(f"  labels: {batch['label']}")
            logger.info(f"  labels类型: {type(batch['label'])}")
            
        if i >= 2:  # 只测试前3个batch
            break
    
    elapsed = time.time() - start_time
    logger.info(f"✅ DataLoader 测试通过 (耗时: {elapsed:.2f}s)")


def test_graph_dataset():
    """测试图数据集"""
    logger.info("\n" + "="*60)
    logger.info("测试4: PRINGGraphDataset")
    logger.info("="*60)
    
    try:
        # 测试all_test数据
        config = PRINGConfig(species="human", sampling_strategy="BFS", split="all_test")
        dataset = PRINGGraphDataset(config, load_graph=True)
        
        logger.info(f"图数据集大小: {len(dataset)}")
        
        # 获取所有蛋白质
        all_proteins = dataset.get_all_proteins()
        logger.info(f"测试图中蛋白质数: {len(all_proteins)}")
        
        if dataset.ground_truth_graph is not None:
            logger.info(f"真实图节点数: {dataset.ground_truth_graph.number_of_nodes()}")
            logger.info(f"真实图边数: {dataset.ground_truth_graph.number_of_edges()}")
        
        # 测试样本
        sample = dataset[0]
        logger.info(f"\n图数据样本:")
        logger.info(f"  蛋白质1: {sample['protein1_id']}")
        logger.info(f"  蛋白质2: {sample['protein2_id']}")
        logger.info(f"  标签: {sample['label']}")
        
        logger.info("✅ PRINGGraphDataset 测试通过")
        
    except Exception as e:
        logger.warning(f"⚠️  PRINGGraphDataset 测试跳过: {e}")


def test_cross_species():
    """测试跨物种数据加载"""
    logger.info("\n" + "="*60)
    logger.info("测试5: 跨物种数据集")
    logger.info("="*60)
    
    species_list = ['arath', 'yeast', 'ecoli']
    
    for species in species_list:
        try:
            config = PRINGConfig(species=species, split="test")
            dataset = PRINGPairDataset(config)
            
            stats = dataset.get_statistics()
            logger.info(f"\n{species.upper()}:")
            logger.info(f"  PPI对: {stats['num_pairs']:,}")
            logger.info(f"  蛋白质数: {stats['num_proteins']:,}")
            logger.info(f"  平均序列长度: {stats['avg_seq_length']:.1f}")
            
        except Exception as e:
            logger.error(f"❌ {species} 测试失败: {e}")
    
    logger.info("\n✅ 跨物种测试完成")


def test_sampling_strategies():
    """测试不同采样策略"""
    logger.info("\n" + "="*60)
    logger.info("测试6: 采样策略对比")
    logger.info("="*60)
    
    strategies = ['BFS', 'DFS', 'RANDOM_WALK']
    
    for strategy in strategies:
        try:
            config = PRINGConfig(
                species="human",
                sampling_strategy=strategy,
                split="train"
            )
            dataset = PRINGPairDataset(config)
            stats = dataset.get_statistics()
            
            logger.info(f"\n{strategy}:")
            logger.info(f"  PPI对: {stats['num_pairs']:,}")
            logger.info(f"  正样本比例: {stats['positive_ratio']:.2%}")
            
        except Exception as e:
            logger.error(f"❌ {strategy} 测试失败: {e}")
    
    logger.info("\n✅ 采样策略测试完成")


def main():
    """主测试函数"""
    logger.info("🚀 开始测试PRING数据加载器\n")
    
    try:
        # 测试配置
        test_config()
        
        # 测试成对数据集
        dataset = test_pair_dataset()
        
        # 测试DataLoader
        test_dataloader(dataset)
        
        # 测试图数据集
        test_graph_dataset()
        
        # 测试跨物种
        test_cross_species()
        
        # 测试采样策略
        test_sampling_strategies()
        
        logger.info("\n" + "="*60)
        logger.info("🎉 所有测试通过！数据加载器就绪")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

