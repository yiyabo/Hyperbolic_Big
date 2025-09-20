#!/usr/bin/env python3
"""
STRING数据库提取器
从STRING v12.0数据库提取置信度>0.95的跨物种蛋白质相互作用数据
"""

import requests
import gzip
from pathlib import Path
import logging
from typing import Dict
from tqdm import tqdm
import sqlite3

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StringDataExtractor:
    """STRING数据提取器"""
    
    def __init__(self, data_dir: str = "data", confidence_threshold: float = 0.95):
        """
        初始化提取器
        
        Args:
            data_dir: 数据存储目录
            confidence_threshold: 置信度阈值
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.confidence_threshold = confidence_threshold
        
        # STRING v12.0 下载URLs
        self.urls = {
            'protein_links': 'https://stringdb-downloads.org/download/protein.links.v12.0.txt.gz',
            'protein_info': 'https://stringdb-downloads.org/download/protein.info.v12.0.txt.gz',
            'protein_sequences': 'https://stringdb-downloads.org/download/protein.sequences.v12.0.fa.gz'
        }
        
        # 数据库文件路径
        self.db_path = self.data_dir / "string_data.db"
        
    def download_file(self, url: str, filename: str) -> Path:
        """下载并解压文件"""
        file_path = self.data_dir / filename
        
        if file_path.exists():
            logger.info("文件 %s 已存在，跳过下载", filename)
            return file_path
            
        logger.info("开始下载 %s...", filename)
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info("下载完成: %s", filename)
        return file_path
    
    def extract_gz_file(self, gz_path: Path) -> Path:
        """解压.gz文件"""
        output_path = gz_path.with_suffix('')
        
        if output_path.exists():
            logger.info("解压文件 %s 已存在，跳过解压", output_path.name)
            return output_path
            
        logger.info("解压文件: %s", gz_path.name)
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                f_out.write(f_in.read())
        
        logger.info("解压完成: %s", output_path.name)
        return output_path
    
    def setup_database(self):
        """创建SQLite数据库和表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建蛋白质信息表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protein_info (
                protein_id TEXT PRIMARY KEY,
                protein_name TEXT,
                species_id INTEGER,
                species_name TEXT,
                annotation TEXT
            )
        ''')
        
        # 创建蛋白质相互作用表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protein_interactions (
                protein1 TEXT,
                protein2 TEXT,
                combined_score INTEGER,
                PRIMARY KEY (protein1, protein2)
            )
        ''')
        
        # 创建蛋白质序列表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS protein_sequences (
                protein_id TEXT PRIMARY KEY,
                sequence TEXT
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_species ON protein_info(species_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_score ON protein_interactions(combined_score)')
        
        conn.commit()
        conn.close()
        logger.info("数据库表结构创建完成")
    
    def load_protein_info(self):
        """加载蛋白质信息到数据库"""
        # 下载并解压蛋白质信息文件
        gz_path = self.download_file(self.urls['protein_info'], 'protein.info.v12.0.txt.gz')
        txt_path = self.extract_gz_file(gz_path)
        
        logger.info("开始加载蛋白质信息...")
        conn = sqlite3.connect(self.db_path)
        
        # 批量插入数据
        chunk_size = 10000
        chunks_processed = 0
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 跳过头部
            next(f)
            
            batch_data = []
            for line in tqdm(f, desc="处理蛋白质信息"):
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    protein_id = parts[0]
                    protein_name = parts[1]
                    species_id = int(protein_id.split('.')[0])
                    annotation = parts[2] if len(parts) > 2 else ""
                    
                    batch_data.append((protein_id, protein_name, species_id, "", annotation))
                    
                    if len(batch_data) >= chunk_size:
                        conn.executemany(
                            'INSERT OR REPLACE INTO protein_info VALUES (?, ?, ?, ?, ?)',
                            batch_data
                        )
                        conn.commit()
                        batch_data = []
                        chunks_processed += 1
            
            # 插入剩余数据
            if batch_data:
                conn.executemany(
                    'INSERT OR REPLACE INTO protein_info VALUES (?, ?, ?, ?, ?)',
                    batch_data
                )
                conn.commit()
        
        conn.close()
        logger.info("蛋白质信息加载完成，处理了 %d 个批次", chunks_processed)
    
    def filter_high_confidence_interactions(self):
        """筛选高置信度的蛋白质相互作用"""
        # 下载并解压相互作用文件
        gz_path = self.download_file(self.urls['protein_links'], 'protein.links.v12.0.txt.gz')
        txt_path = self.extract_gz_file(gz_path)
        
        logger.info("开始筛选置信度 > %.2f 的相互作用...", self.confidence_threshold)
        conn = sqlite3.connect(self.db_path)
        
        chunk_size = 10000
        high_confidence_count = 0
        total_count = 0
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            # 跳过头部
            next(f)
            
            batch_data = []
            for line in tqdm(f, desc="筛选高置信度相互作用"):
                parts = line.strip().split()
                if len(parts) >= 3:
                    protein1 = parts[0]
                    protein2 = parts[1]
                    combined_score = int(parts[2])
                    
                    total_count += 1
                    
                    # 只保留高置信度的相互作用
                    if combined_score >= self.confidence_threshold * 1000:  # STRING分数是0-1000
                        batch_data.append((protein1, protein2, combined_score))
                        high_confidence_count += 1
                        
                        if len(batch_data) >= chunk_size:
                            conn.executemany(
                                'INSERT OR REPLACE INTO protein_interactions VALUES (?, ?, ?)',
                                batch_data
                            )
                            conn.commit()
                            batch_data = []
            
            # 插入剩余数据
            if batch_data:
                conn.executemany(
                    'INSERT OR REPLACE INTO protein_interactions VALUES (?, ?, ?)',
                    batch_data
                )
                conn.commit()
        
        conn.close()
        logger.info("相互作用筛选完成：%d/%d (%.2f%%) 符合阈值", 
                   high_confidence_count, total_count, high_confidence_count/total_count*100)
    
    def load_protein_sequences(self):
        """加载蛋白质序列"""
        # 下载并解压序列文件
        gz_path = self.download_file(self.urls['protein_sequences'], 'protein.sequences.v12.0.fa.gz')
        
        logger.info("开始加载蛋白质序列...")
        conn = sqlite3.connect(self.db_path)
        
        chunk_size = 1000
        batch_data = []
        
        with gzip.open(gz_path, 'rt') as f:
            protein_id = None
            sequence = []
            
            for line in tqdm(f, desc="处理蛋白质序列"):
                line = line.strip()
                if line.startswith('>'):
                    # 保存前一个蛋白质的序列
                    if protein_id and sequence:
                        batch_data.append((protein_id, ''.join(sequence)))
                        
                        if len(batch_data) >= chunk_size:
                            conn.executemany(
                                'INSERT OR REPLACE INTO protein_sequences VALUES (?, ?)',
                                batch_data
                            )
                            conn.commit()
                            batch_data = []
                    
                    # 解析新的蛋白质ID
                    protein_id = line[1:].split()[0]
                    sequence = []
                else:
                    sequence.append(line)
            
            # 处理最后一个蛋白质
            if protein_id and sequence:
                batch_data.append((protein_id, ''.join(sequence)))
            
            # 插入剩余数据
            if batch_data:
                conn.executemany(
                    'INSERT OR REPLACE INTO protein_sequences VALUES (?, ?)',
                    batch_data
                )
                conn.commit()
        
        conn.close()
        logger.info("蛋白质序列加载完成")
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 统计信息
        stats = {}
        
        # 蛋白质总数
        cursor.execute('SELECT COUNT(*) FROM protein_info')
        stats['total_proteins'] = cursor.fetchone()[0]
        
        # 物种数量
        cursor.execute('SELECT COUNT(DISTINCT species_id) FROM protein_info')
        stats['total_species'] = cursor.fetchone()[0]
        
        # 高置信度相互作用数量
        cursor.execute('SELECT COUNT(*) FROM protein_interactions')
        stats['high_confidence_interactions'] = cursor.fetchone()[0]
        
        # 有序列的蛋白质数量
        cursor.execute('SELECT COUNT(*) FROM protein_sequences')
        stats['proteins_with_sequences'] = cursor.fetchone()[0]
        
        # 各物种的蛋白质数量（前10）
        cursor.execute('''
            SELECT species_id, COUNT(*) as count 
            FROM protein_info 
            GROUP BY species_id 
            ORDER BY count DESC 
            LIMIT 10
        ''')
        stats['top_species'] = cursor.fetchall()
        
        conn.close()
        return stats
    
    def extract_all_data(self):
        """执行完整的数据提取流程"""
        logger.info("开始STRING数据提取流程...")
        
        # 1. 设置数据库
        self.setup_database()
        
        # 2. 加载蛋白质信息
        self.load_protein_info()
        
        # 3. 筛选高置信度相互作用
        self.filter_high_confidence_interactions()
        
        # 4. 加载蛋白质序列
        self.load_protein_sequences()
        
        # 5. 输出统计信息
        stats = self.get_statistics()
        logger.info("数据提取完成！统计信息:")
        for key, value in stats.items():
            logger.info("  %s: %s", key, value)
        
        return stats

def main():
    """主函数"""
    extractor = StringDataExtractor(confidence_threshold=0.95)
    stats = extractor.extract_all_data()
    
    print("\n" + "="*50)
    print("STRING数据提取完成！")
    print("="*50)
    print(f"总蛋白质数量: {stats['total_proteins']:,}")
    print(f"总物种数量: {stats['total_species']:,}")
    print(f"高置信度相互作用: {stats['high_confidence_interactions']:,}")
    print(f"有序列的蛋白质: {stats['proteins_with_sequences']:,}")
    print("\n前10个物种的蛋白质数量:")
    for species_id, count in stats['top_species']:
        print(f"  物种 {species_id}: {count:,} 个蛋白质")

if __name__ == "__main__":
    main()
