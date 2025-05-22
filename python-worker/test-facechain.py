#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试FaceChain集成功能
"""

import os
import sys
import logging
from pathlib import Path
import shutil
import uuid
import json

# 配置日志 - 调整为INFO级别确保可以看到所有输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # 明确指定输出到stdout
)
logger = logging.getLogger('TestFaceChain')

# 确保显示其他模块的日志
logging.getLogger('FaceChainIntegration').setLevel(logging.DEBUG)

print("开始测试FaceChain集成功能...")

# 导入FaceChain集成模块
try:
    from facechain_integration import facechain_generator
    print("成功导入FaceChain集成模块")
except Exception as e:
    print(f"导入FaceChain集成模块失败: {e}")
    sys.exit(1)

def main():
    """主函数"""
    print("=" * 50)
    logger.info("开始测试FaceChain集成功能")
    
    # 基础路径
    base_dir = Path(__file__).parent.parent
    print(f"基础路径: {base_dir}")
    
    # 创建测试任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"测试任务ID: {task_id}")
    
    # 查找现有的上传目录，使用第一个找到的目录
    upload_dirs = list(base_dir.glob('data/uploads/*'))
    if not upload_dirs:
        logger.error("未找到上传目录，无法进行测试")
        return 1
    
    # 使用第一个目录
    sample_dir = upload_dirs[0]
    logger.info(f"使用样本目录: {sample_dir}")
    
    # 收集照片
    photos = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        photos.extend([str(p) for p in sample_dir.glob(ext)])
    
    if not photos:
        logger.error("未找到样本照片，无法进行测试")
        return 1
    
    logger.info(f"找到 {len(photos)} 张样本照片")
    
    # 创建输出目录
    output_dir = base_dir / 'data' / 'outputs' / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")
    
    # 只测试一种风格，减少测试时间
    styles = ["油画"]
    
    # 分别测试每种风格
    for style in styles:
        style_output_dir = output_dir / style
        style_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"测试风格: {style}")
        print(f"开始生成 '{style}' 风格的照片")
        
        try:
            result = facechain_generator.generate(
                task_id, 
                photos, 
                [style],
                style_output_dir
            )
            
            if result:
                logger.info(f"风格 {style} 测试成功")
                # 列出生成的文件
                generated_files = list(style_output_dir.glob('*'))
                logger.info(f"生成了 {len(generated_files)} 个文件: {[f.name for f in generated_files]}")
            else:
                logger.error(f"风格 {style} 测试失败")
        except Exception as e:
            logger.error(f"生成过程发生异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    logger.info(f"测试完成，结果保存在 {output_dir}")
    logger.info(f"请查看生成的照片质量和风格是否正确")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"测试过程发生未捕获异常: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1) 