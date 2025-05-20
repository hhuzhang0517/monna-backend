#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceChain模拟示例生成器
使用PIL库进行基本图像处理，模拟FaceChain的效果
"""

import os
import sys
import logging
import random
import uuid
from pathlib import Path

# 尝试导入PIL
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
except ImportError:
    print("请先安装Pillow和NumPy: pip install pillow numpy")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('SampleGenerator')

def apply_style(img, style):
    """
    应用风格到图片
    
    Args:
        img: PIL图像对象
        style: 风格名称("婚纱", "油画", "古风", "漫画")
        
    Returns:
        PIL图像对象
    """
    # 复制图像避免修改原始图像
    img = img.copy()
    
    # 根据风格应用不同效果
    if style == "婚纱":
        # 明亮、柔和的婚纱风格
        img = ImageEnhance.Brightness(img).enhance(1.1)
        img = ImageEnhance.Contrast(img).enhance(0.9)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        # 添加一些明亮的色调
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.2)
    elif style == "油画":
        # 油画效果
        img = img.filter(ImageFilter.EDGE_ENHANCE)
        img = img.filter(ImageFilter.GaussianBlur(radius=1))
        # 增加饱和度
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.5)
    elif style == "古风":
        # 古风效果，偏棕褐色
        img = ImageOps.grayscale(img)
        img = ImageOps.colorize(img, "#8B4513", "#F5DEB3")
    elif style == "漫画":
        # 漫画风格，提高对比度，简化色彩
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = ImageEnhance.Color(img).enhance(1.5)
        img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    else:
        # 默认随机效果
        # 随机调整亮度、对比度、饱和度
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.9, 1.3))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.9, 1.3))
        
        # 随机应用滤镜
        filters = [
            ImageFilter.CONTOUR,
            ImageFilter.EDGE_ENHANCE,
            ImageFilter.EMBOSS,
            ImageFilter.SMOOTH,
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5))
        ]
        img = img.filter(random.choice(filters))
    
    return img

def add_variations(base_img, variations=5):
    """
    基于基础图像创建多个变体
    
    Args:
        base_img: 基础PIL图像
        variations: 要创建的变体数量
        
    Returns:
        变体图像列表
    """
    results = []
    for i in range(variations):
        # 创建轻微变化的图像
        img = base_img.copy()
        
        # 随机调整参数
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)
        saturation = random.uniform(0.9, 1.1)
        
        # 应用调整
        img = ImageEnhance.Brightness(img).enhance(brightness)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(saturation)
        
        # 应用轻微模糊或锐化
        if random.choice([True, False]):
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.7)))
        else:
            img = img.filter(ImageFilter.SHARPEN)
        
        results.append(img)
    
    return results

def generate_samples(input_photos, styles, output_dir):
    """
    为每种风格生成样本图像
    
    Args:
        input_photos: 输入照片路径列表
        styles: 风格列表
        output_dir: 输出目录
    """
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 为每种风格创建目录
    for style in styles:
        style_dir = output_dir / style
        style_dir.mkdir(exist_ok=True)
        
        logger.info(f"正在生成'{style}'风格的样本图像...")
        
        # 为每张输入照片生成风格化变体
        for photo_path in input_photos:
            try:
                photo_path = Path(photo_path)
                if not photo_path.exists():
                    logger.warning(f"照片不存在: {photo_path}")
                    continue
                
                # 打开输入图像
                img = Image.open(photo_path)
                
                # 应用风格
                styled_img = apply_style(img, style)
                
                # 创建变体
                variants = add_variations(styled_img)
                
                # 保存结果
                for i, variant in enumerate(variants):
                    output_file = style_dir / f"{photo_path.stem}_variant_{i+1}.jpg"
                    variant.save(output_file, quality=95)
                    logger.info(f"创建样本: {output_file}")
            except Exception as e:
                logger.error(f"处理照片 {photo_path} 失败: {e}")
    
    logger.info(f"样本生成完成，结果保存在: {output_dir}")

def main():
    """主函数"""
    logger.info("开始生成样本图像")
    
    # 基础路径
    base_dir = Path(__file__).parent.parent
    
    # 查找现有的上传目录，使用第一个找到的目录
    upload_dirs = list(base_dir.glob('data/uploads/*'))
    if not upload_dirs:
        logger.error("未找到上传目录，无法进行样本生成")
        return 1
    
    # 使用第一个目录
    sample_dir = upload_dirs[0]
    logger.info(f"使用样本目录: {sample_dir}")
    
    # 收集照片
    photos = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        photos.extend([str(p) for p in sample_dir.glob(ext)])
    
    if not photos:
        logger.error("未找到样本照片，无法进行样本生成")
        return 1
    
    logger.info(f"找到 {len(photos)} 张样本照片")
    
    # 创建输出目录
    task_id = str(uuid.uuid4())
    output_dir = base_dir / 'data' / 'outputs' / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建输出目录: {output_dir}")
    
    # 定义要生成的风格
    styles = ["婚纱", "油画", "古风", "漫画"]
    
    # 生成样本图像
    generate_samples(photos, styles, output_dir)
    
    logger.info(f"样本生成完成，结果保存在: {output_dir}")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"样本生成过程中发生未捕获异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1) 