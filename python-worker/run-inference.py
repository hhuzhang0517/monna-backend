#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceChain推理脚本
这个脚本是从FaceChain整合到Monna项目的简化版本
"""

import os
import sys
import json
import argparse
from pathlib import Path

# 添加FaceChain路径
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
FACECHAIN_DIR = MODELS_DIR / 'facechain'
sys.path.append(str(FACECHAIN_DIR))

try:
    from facechain.inference_fact import GenPortrait
    print("成功导入FaceChain核心模块")
except ImportError as e:
    print(f"导入FaceChain核心模块失败: {e}")
    print("请确保已正确安装FaceChain及其依赖")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='FaceChain推理')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    print(f"FaceChain推理开始...")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {args.output_dir}")
    
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 获取配置参数
    input_paths = config.get("input_img_paths", [])
    if not input_paths:
        print("错误: 没有找到输入图像路径")
        return 1
    
    style = config.get("style", "portrait")
    use_pose_model = config.get("use_pose_model", False)
    pose_image = config.get("pose_image", None)
    num_generate = config.get("num_generate", 5)
    multiplier_style = config.get("multiplier_style", 0.25)
    
    print(f"输入图像数量: {len(input_paths)}")
    print(f"使用风格: {style}")
    print(f"生成数量: {num_generate}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 创建生成器实例
        generator = GenPortrait(
            style,
            multiplier_style,
            num_inference_steps=50,
            negative_prompt=None,
            super_resolution=True,
            use_pose=use_pose_model,
            pose_image=pose_image,
            random_pose=not use_pose_model,
        )
        
        # 生成肖像
        output_images = generator.generate_portrait(
            input_paths,
            args.output_dir,
            num_generate
        )
        
        print(f"FaceChain推理完成，生成了 {len(output_images)} 张图像")
        print(f"结果保存到: {args.output_dir}")
        return 0
    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 