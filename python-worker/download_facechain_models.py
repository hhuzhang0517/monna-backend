#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载FaceChain所需的模型
"""

import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FaceChainModels')

# 添加FaceChain路径
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
FACECHAIN_DIR = MODELS_DIR / 'facechain'

# 尝试导入modelscope
try:
    from modelscope import snapshot_download
    logger.info("成功导入ModelScope")
except ImportError:
    logger.error("无法导入ModelScope，请确保已安装: pip install modelscope")
    sys.exit(1)

def download_model(model_id):
    """下载指定模型"""
    logger.info(f"正在下载模型: {model_id}")
    try:
        model_dir = snapshot_download(model_id)
        logger.info(f"模型下载成功: {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"下载模型 {model_id} 失败: {e}")
        return None

def main():
    """下载FaceChain所需的所有模型"""
    # FaceChain所需的所有模型ID
    models = [
        'iic/cv_vit_face-recognition',
        'damo/cv_ddsar_face-detection_iclr23-damofd',
        'damo/cv_resnet101_image-multiple-human-parsing',
        'damo/cv_unet_skin_retouching_torch',
        'damo/cv_unet_face_fusion_torch',
        'yucheng1996/FaceChain-FACT',
        'damo/cv_resnet34_face-attribute-recognition_fairface'
    ]
    
    # 下载所有模型
    successful = 0
    for model_id in models:
        if download_model(model_id):
            successful += 1
    
    # 显示下载结果
    logger.info(f"模型下载完成: {successful}/{len(models)} 成功")

if __name__ == "__main__":
    main() 