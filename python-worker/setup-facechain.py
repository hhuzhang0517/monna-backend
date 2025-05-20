#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FaceChain 安装脚本
这个脚本会克隆FaceChain仓库，并安装所有必要的依赖和模型
"""

import os
import sys
import platform
import subprocess
import logging
from pathlib import Path
import tempfile
import shutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('FaceChainSetup')

# 配置路径
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
FACECHAIN_DIR = MODELS_DIR / 'facechain'

def run_command(cmd, cwd=None, check=True):
    """运行命令并记录输出"""
    logger.info(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            cwd=cwd,
            capture_output=True,
            text=True
        )
        if result.stdout:
            logger.info(f"命令输出: {result.stdout}")
        if result.stderr and not result.returncode == 0:
            logger.warning(f"命令警告: {result.stderr}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令执行失败: {e}")
        if e.stdout:
            logger.info(f"输出: {e.stdout}")
        if e.stderr:
            logger.error(f"错误: {e.stderr}")
        if check:
            raise
        return e

def clone_facechain():
    """克隆FaceChain仓库"""
    if FACECHAIN_DIR.exists():
        logger.info(f"FaceChain仓库已存在: {FACECHAIN_DIR}")
        # 更新仓库
        logger.info("正在更新FaceChain仓库...")
        run_command(
            ["git", "pull"],
            cwd=str(FACECHAIN_DIR),
            check=False
        )
        return
    
    logger.info(f"正在克隆FaceChain仓库到: {FACECHAIN_DIR}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "git", "clone", 
            "https://github.com/modelscope/facechain.git",
            str(FACECHAIN_DIR)
        ],
        cwd=str(MODELS_DIR)
    )

def install_dependencies():
    """安装FaceChain所需的所有依赖"""
    logger.info("安装FaceChain所需的依赖...")
    
    # 基础依赖
    basic_deps = [
        "torch>=2.0.0", 
        "torchvision", 
        "diffusers>=0.14.0", 
        "transformers>=4.28.1", 
        "accelerate>=0.16.0",
        "controlnet_aux", 
        "onnxruntime",
        "opencv-python",
        "modelscope",
        "matplotlib",
        "pillow",
        "scikit-image"
    ]
    
    logger.info("安装基础依赖...")
    for dep in basic_deps:
        try:
            run_command(["pip", "install", dep])
        except Exception:
            logger.warning(f"安装依赖 {dep} 失败，但将继续安装其他依赖")
    
    # 安装FaceChain的requirements.txt
    req_file = FACECHAIN_DIR / "requirements.txt"
    if req_file.exists():
        logger.info(f"使用FaceChain的requirements.txt安装依赖: {req_file}")
        try:
            run_command(["pip", "install", "-r", str(req_file)])
        except Exception:
            logger.warning("安装requirements.txt中的依赖失败，但将继续安装")

def download_models():
    """下载FaceChain所需的模型"""
    logger.info("下载并准备FaceChain所需的模型...")
    
    # 使用FaceChain的install.py来下载模型
    install_script = FACECHAIN_DIR / "install.py"
    if install_script.exists():
        logger.info("使用FaceChain的install.py脚本下载模型...")
        try:
            # 先尝试直接运行install.py
            run_command(["python", str(install_script)], cwd=str(FACECHAIN_DIR))
        except Exception:
            logger.warning("直接运行install.py失败，尝试替代方法...")
            try:
                # 将FaceChain目录添加到Python路径
                logger.info("将FaceChain目录添加到Python路径...")
                sys.path.append(str(FACECHAIN_DIR))
                
                # 使用modelscope下载模型
                logger.info("使用modelscope下载模型...")
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('iic/cv_vit_face-recognition')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('damo/cv_ddsar_face-detection_iclr23-damofd')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('damo/cv_resnet101_image-multiple-human-parsing')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('damo/cv_unet_skin_retouching_torch')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('damo/cv_unet_face_fusion_torch')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('yucheng1996/FaceChain-FACT')"
                ])
                run_command([
                    "python", "-c", 
                    "from modelscope import snapshot_download; "
                    "snapshot_download('damo/cv_resnet34_face-attribute-recognition_fairface')"
                ])
            except Exception as e:
                logger.error(f"下载模型失败: {e}")
                logger.warning("模型下载失败，但将继续安装其他组件")

def configure_facechain():
    """配置FaceChain以便与Monna集成"""
    logger.info("配置FaceChain集成...")
    
    # 确保run_inference.py存在并且可用
    run_inference = FACECHAIN_DIR / "run_inference.py"
    if not run_inference.exists():
        logger.warning(f"未找到run_inference.py: {run_inference}")
        # 从FaceChain仓库复制样例代码
        sample_inference = FACECHAIN_DIR / "scripts" / "run_inference.py"
        if sample_inference.exists():
            logger.info(f"从样例复制run_inference.py: {sample_inference} -> {run_inference}")
            shutil.copy(sample_inference, run_inference)
        else:
            logger.warning("未找到样例run_inference.py，创建一个基本版本...")
            with open(run_inference, 'w') as f:
                f.write("""
import os
import argparse
import json
from facechain.inference_fact import GenPortrait

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"FaceChain推理开始...")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {args.output_dir}")
    
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # 获取输入图像路径
    input_paths = config.get("input_img_paths", [])
    if not input_paths and isinstance(input_paths, list):
        print("错误: 没有找到输入图像路径")
        return 1
    
    # 获取风格和其他参数
    style = config.get("style", "portrait")
    use_pose_model = config.get("use_pose_model", False)
    pose_image = config.get("pose_image", None)
    num_generate = config.get("num_generate", 5)
    multiplier_style = config.get("multiplier_style", 0.25)
    
    print(f"使用风格: {style}")
    print(f"生成数量: {num_generate}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 调用FaceChain生成肖像
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
    
    generator.generate_portrait(
        input_paths,
        args.output_dir,
        num_generate
    )
    
    print(f"FaceChain推理完成，结果保存到: {args.output_dir}")
    return 0

if __name__ == "__main__":
    main()
""")
    
    logger.info("FaceChain配置完成")

def main():
    logger.info("开始安装并配置FaceChain...")
    
    # 1. 克隆FaceChain仓库
    clone_facechain()
    
    # 2. 安装依赖
    install_dependencies()
    
    # 3. 下载模型
    download_models()
    
    # 4. 配置FaceChain
    configure_facechain()
    
    logger.info("FaceChain安装和配置完成！")
    logger.info(f"FaceChain位置: {FACECHAIN_DIR}")
    
    # 导入测试
    try:
        sys.path.append(str(FACECHAIN_DIR))
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
        
        # 尝试导入FaceChain核心模块
        try:
            from facechain.inference_fact import GenPortrait
            logger.info("成功导入FaceChain核心模块 - 安装完成！")
        except ImportError as e:
            logger.warning(f"无法导入FaceChain核心模块: {e}")
            logger.warning("部分组件可能未正确安装，但基本安装已完成")
    except ImportError as e:
        logger.error(f"导入测试失败: {e}")
    
    logger.info("FaceChain安装脚本执行完毕")

if __name__ == "__main__":
    main() 