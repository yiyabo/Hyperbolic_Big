#!/usr/bin/env python3
"""
测试STRING数据提取器的简单脚本
"""

import sys
from pathlib import Path

# 添加data_extraction目录到路径
sys.path.append(str(Path(__file__).parent / "data_extraction"))

from string_data_extractor import StringDataExtractor

def test_extractor():
    """测试数据提取器"""
    print("开始测试STRING数据提取器...")
    
    # 创建提取器实例
    extractor = StringDataExtractor(confidence_threshold=0.95)
    
    print(f"数据目录: {extractor.data_dir}")
    print(f"置信度阈值: {extractor.confidence_threshold}")
    print(f"数据库路径: {extractor.db_path}")
    
    # 检查URL可访问性
    print("\n检查下载URL:")
    for name, url in extractor.urls.items():
        print(f"  {name}: {url}")
    
    print("\n如果要开始完整的数据提取，请运行:")
    print("python data_extraction/string_data_extractor.py")
    
    return True

if __name__ == "__main__":
    test_extractor()
