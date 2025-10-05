#!/usr/bin/env python3
"""
PRING数据加载示例

展示如何使用数据加载器进行基本的数据加载和处理
"""

import sys
sys.path.append('..')

from data_loader import PRINGPairDataset, PRINGConfig, get_dataloader


def example_basic_usage():
    """示例1: 基本使用"""
    print("="*60)
    print("示例1: 基本数据加载")
    print("="*60)
    
    # 创建配置
    config = PRINGConfig(
        species="human",
        sampling_strategy="BFS",
        split="train"
    )
    
    # 创建数据集
    dataset = PRINGPairDataset(config)
    
    # 查看统计信息
    stats = dataset.get_statistics()
    print(f"\n数据集统计:")
    print(f"  总PPI对: {stats['num_pairs']:,}")
    print(f"  蛋白质数: {stats['num_proteins']:,}")
    print(f"  正样本比例: {stats['positive_ratio']:.2%}")
    
    # 访问单个样本
    sample = dataset[0]
    print(f"\n第一个样本:")
    print(f"  序列1长度: {len(sample['seq1'])}")
    print(f"  序列2长度: {len(sample['seq2'])}")
    print(f"  标签: {sample['label']}")


def example_with_dataloader():
    """示例2: 使用DataLoader进行批处理"""
    print("\n" + "="*60)
    print("示例2: 批处理数据加载")
    print("="*60)
    
    # 创建数据集
    config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
    dataset = PRINGPairDataset(config, return_ids=True)
    
    # 创建DataLoader
    dataloader = get_dataloader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    
    print(f"\nDataLoader信息:")
    print(f"  Batch数量: {len(dataloader)}")
    print(f"  Batch大小: {dataloader.batch_size}")
    
    # 遍历一个batch
    for batch in dataloader:
        print(f"\nBatch内容:")
        print(f"  序列1: {len(batch['seq1'])} 条")
        print(f"  序列2: {len(batch['seq2'])} 条")
        print(f"  标签: {batch['label']}")
        print(f"  蛋白质ID示例: {batch['protein1_id'][0]}, {batch['protein2_id'][0]}")
        break  # 只展示第一个batch


def example_train_val_test():
    """示例3: 完整的训练/验证/测试流程"""
    print("\n" + "="*60)
    print("示例3: 训练/验证/测试数据集")
    print("="*60)
    
    # 训练集
    train_config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
    train_dataset = PRINGPairDataset(train_config)
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 验证集
    val_config = PRINGConfig(species="human", sampling_strategy="BFS", split="val")
    val_dataset = PRINGPairDataset(val_config)
    val_loader = get_dataloader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # 测试集
    test_config = PRINGConfig(species="human", sampling_strategy="BFS", split="test")
    test_dataset = PRINGPairDataset(test_config)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_dataset):,} 样本, {len(train_loader)} batches")
    print(f"  验证集: {len(val_dataset):,} 样本, {len(val_loader)} batches")
    print(f"  测试集: {len(test_dataset):,} 样本, {len(test_loader)} batches")
    
    # 模拟训练循环
    print(f"\n模拟训练循环:")
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练
        for i, batch in enumerate(train_loader):
            if i >= 2:  # 只展示前2个batch
                break
            print(f"  训练 batch {i+1}: {len(batch['label'])} 样本")
        
        # 验证
        for i, batch in enumerate(val_loader):
            if i >= 1:  # 只展示第1个batch
                break
            print(f"  验证 batch {i+1}: {len(batch['label'])} 样本")


def example_cross_species():
    """示例4: 跨物种测试"""
    print("\n" + "="*60)
    print("示例4: 跨物种泛化测试")
    print("="*60)
    
    print("\n在人类数据上训练后，在其他物种上测试:")
    
    for species in ['arath', 'yeast', 'ecoli']:
        config = PRINGConfig(species=species, split="test")
        dataset = PRINGPairDataset(config)
        
        stats = dataset.get_statistics()
        print(f"\n{species.upper()}:")
        print(f"  测试样本: {stats['num_pairs']:,}")
        print(f"  蛋白质数: {stats['num_proteins']:,}")


def example_with_transform():
    """示例5: 使用自定义转换函数"""
    print("\n" + "="*60)
    print("示例5: 自定义序列转换")
    print("="*60)
    
    def simple_tokenize(seq):
        """简单的tokenization示例"""
        # 这里可以集成ESM tokenizer
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_to_id = {aa: i for i, aa in enumerate(amino_acids)}
        return [aa_to_id.get(aa, 20) for aa in seq]  # 20为未知氨基酸
    
    # 使用转换函数
    config = PRINGConfig(species="human", sampling_strategy="BFS", split="train")
    dataset = PRINGPairDataset(config, transform=simple_tokenize)
    
    sample = dataset[0]
    print(f"\n转换后的样本:")
    print(f"  原始序列类型: str")
    print(f"  转换后类型: {type(sample['seq1'])}")
    print(f"  转换后长度: {len(sample['seq1'])}")
    print(f"  前10个token: {sample['seq1'][:10]}")


def main():
    """运行所有示例"""
    print("🚀 PRING数据加载器使用示例\n")
    
    # 运行示例
    example_basic_usage()
    example_with_dataloader()
    example_train_val_test()
    example_cross_species()
    example_with_transform()
    
    print("\n" + "="*60)
    print("✅ 所有示例运行完成！")
    print("="*60)
    print("\n提示:")
    print("  1. 查看 data_loader/README.md 了解更多用法")
    print("  2. 运行 python test_data_loader.py 进行完整测试")
    print("  3. 参考 docs/pring_dataset.md 了解数据集详情")


if __name__ == "__main__":
    main()

