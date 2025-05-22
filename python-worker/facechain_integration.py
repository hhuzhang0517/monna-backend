import os
import sys
import logging
from pathlib import Path
import subprocess
import tempfile
import json
import shutil
import time
import random
from io import BytesIO

# 新增导入图像处理库
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logging.getLogger('FaceChainIntegration').warning("PIL或numpy未安装，高级图像处理将不可用")

logger = logging.getLogger('FaceChainIntegration')

# 配置路径
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
FACECHAIN_DIR = MODELS_DIR / 'facechain'

class FaceChainGenerator:
    def __init__(self):
        """初始化FaceChain生成器"""
        self.initialized = False
        self.facechain_available = False
        self.initialize()
    
    def initialize(self):
        """初始化FaceChain环境"""
        try:
            # 确保FaceChain目录存在
            if not FACECHAIN_DIR.exists():
                logger.info(f"FaceChain目录不存在，将运行安装脚本...")
                self._setup_facechain()
            
            # 设置Python环境
            sys.path.append(str(FACECHAIN_DIR))
            
            # 尝试导入PyTorch检查环境
            try:
                import torch
                logger.info(f"PyTorch版本: {torch.__version__}")
                logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
                if torch.cuda.is_available():
                    logger.info(f"当前CUDA设备: {torch.cuda.get_device_name(0)}")
            except ImportError:
                logger.warning("PyTorch未安装，推理性能可能受限")
            
            # 验证FaceChain可用性
            if self._check_facechain_availability():
                logger.info("FaceChain模型和依赖已就绪")
                self.facechain_available = True
            else:
                logger.warning("FaceChain组件未完全安装，将使用模拟模式")
                self._create_mock_inference()
            
            self.initialized = True
            logger.info("FaceChain初始化完成")
            
        except Exception as e:
            logger.error(f"初始化FaceChain失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _setup_facechain(self):
        """安装FaceChain"""
        setup_script = Path(__file__).parent / "setup-facechain.py"
        if setup_script.exists():
            logger.info(f"运行FaceChain安装脚本: {setup_script}")
            try:
                subprocess.run(
                    ["python", str(setup_script)],
                    check=True,
                    cwd=str(Path(__file__).parent)
                )
                logger.info("FaceChain安装完成")
            except subprocess.CalledProcessError as e:
                logger.error(f"安装FaceChain失败: {e}")
                raise
        else:
            logger.error(f"未找到FaceChain安装脚本: {setup_script}")
            raise FileNotFoundError(f"未找到FaceChain安装脚本: {setup_script}")
    
    def _check_facechain_availability(self):
        """检查FaceChain是否可用"""
        # 首先尝试用我们的自定义推理脚本
        custom_inference_script = Path(__file__).parent / "run-inference.py"
        if custom_inference_script.exists():
            logger.info(f"找到自定义推理脚本: {custom_inference_script}")
            
            # 尝试导入核心模块
            try:
                sys.path.append(str(FACECHAIN_DIR))
                from facechain.inference_fact import GenPortrait
                logger.info("成功导入FaceChain核心模块")
                return True
            except ImportError as e:
                logger.warning(f"无法导入FaceChain核心模块: {e}")
                return False
        else:
            # 如果自定义脚本不存在，检查FaceChain原生脚本
            inference_script = FACECHAIN_DIR / "run_inference.py"
            if not inference_script.exists():
                logger.warning(f"未找到run_inference.py: {inference_script}")
                return False
            
            # 尝试导入核心模块
            try:
                sys.path.append(str(FACECHAIN_DIR))
                from facechain.inference_fact import GenPortrait
                logger.info("成功导入FaceChain核心模块")
                return True
            except ImportError as e:
                logger.warning(f"无法导入FaceChain核心模块: {e}")
                return False
    
    def _create_mock_inference(self):
        """创建模拟推理脚本（当FaceChain不可用时）"""
        logger.info("创建模拟推理脚本...")
        
        # 创建模拟run_inference.py脚本
        mock_inference = FACECHAIN_DIR / "mock_inference.py"
        with open(mock_inference, 'w') as f:
            f.write('''
import os
import sys
import shutil
import json
import argparse
import random
from pathlib import Path

try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("警告: PIL或numpy未安装，将使用基本图像复制")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    return parser.parse_args()

def apply_random_style(img_path, output_path, style=None):
    """应用随机风格效果到图片上，模拟AI生成效果"""
    if not HAS_PIL:
        # 如果PIL不可用，直接复制
        shutil.copy(img_path, output_path)
        return
    
    try:
        # 打开图片
        img = Image.open(img_path)
        
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
        
        # 保存处理后的图片
        img.save(output_path)
    except Exception as e:
        print(f"图像处理失败: {e}")
        # 出错时回退到直接复制
        shutil.copy(img_path, output_path)

def main():
    args = parse_args()
    print(f"模拟FaceChain推理开始...")
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
    
    # 获取风格
    style = config.get("style", "portrait")
    print(f"应用风格: {style}")
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 模拟生成5张照片
    for i in range(5):
        # 随机选择一张输入图像作为基础
        input_path = random.choice(input_paths) if len(input_paths) > 1 else input_paths[0]
        output_file = output_dir / f"result_{i+1}.jpg"
        
        if os.path.exists(input_path):
            print(f"处理照片 {i+1}/5: 应用 {style} 风格到 {input_path}")
            # 应用随机效果
            apply_random_style(input_path, output_file, style)
            print(f"生成结果保存至: {output_file}")
        else:
            # 创建空白图片文件
            with open(output_file, 'w') as f:
                f.write('Simulated AI photo')
            print(f"警告: 输入路径 {input_path} 不存在，创建空白图片")
    
    print(f"模拟FaceChain推理完成，生成了5张照片")
    return 0

if __name__ == "__main__":
    sys.exit(main())
''')
        logger.info(f"模拟推理脚本创建完成: {mock_inference}")
    
    def generate(self, task_id, photos, styles, output_dir):
        """
        生成AI人像照片
        
        Args:
            task_id: 任务ID
            photos: 照片路径列表
            styles: 风格列表
            output_dir: 输出目录
        
        Returns:
            bool: 是否成功
        """
        if not self.initialized:
            logger.error("FaceChain未初始化，无法生成照片")
            return False
        
        try:
            logger.info(f"开始生成任务 {task_id} 的AI照片")
            
            # 准备配置文件
            config_file = self._prepare_config(task_id, photos, styles)
            
            # 首先尝试使用自定义推理脚本
            custom_inference_script = Path(__file__).parent / "run-inference.py"
            if custom_inference_script.exists() and self.facechain_available:
                inference_script = str(custom_inference_script)
                logger.info("使用自定义FaceChain推理脚本")
                
                # 使用自定义推理脚本的命令
                cmd = [
                    "python", 
                    inference_script,
                    "--config", str(config_file),
                    "--output_dir", str(output_dir)
                ]
                
                logger.info(f"执行命令: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                logger.info(f"FaceChain生成完成，输出: {result.stdout}")
                if result.stderr:
                    logger.warning(f"FaceChain警告: {result.stderr}")
                
                return True
            
            # 如果自定义脚本不可用或失败，尝试使用FaceChain原生脚本
            elif self.facechain_available:
                inference_script = "run_inference.py"
                logger.info("使用FaceChain原生推理脚本")
                
                # 调用推理脚本
                cmd = [
                    "python", 
                    inference_script,
                    "--config", str(config_file),
                    "--output_dir", str(output_dir)
                ]
                
                logger.info(f"执行命令: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(FACECHAIN_DIR),
                    capture_output=True,
                    text=True
                )
                
                logger.info(f"FaceChain生成完成，输出: {result.stdout}")
                if result.stderr:
                    logger.warning(f"FaceChain警告: {result.stderr}")
                
                return True
            else:
                # 如果FaceChain不可用，使用模拟模式
                logger.warning("FaceChain不可用，使用模拟模式")
                return self._simulate_generation(task_id, photos, styles, output_dir)
                
        except subprocess.CalledProcessError as e:
            logger.error(f"调用FaceChain失败: {e}")
            logger.error(f"错误输出: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            
            # 如果真实推理失败，尝试使用模拟模式
            logger.warning("FaceChain推理失败，切换到模拟模式")
            return self._simulate_generation(task_id, photos, styles, output_dir)
        except Exception as e:
            logger.error(f"生成AI照片时发生错误: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _simulate_generation(self, task_id, photos, styles, output_dir):
        """使用PIL模拟生成照片（退化方案）"""
        logger.warning("使用PIL模拟生成照片")
        
        try:
            # 使用mock_inference.py脚本
            mock_script = FACECHAIN_DIR / "mock_inference.py"
            if not mock_script.exists():
                self._create_mock_inference()
            
            # 准备配置文件
            config_file = self._prepare_config(task_id, photos, styles)
            
            # 调用模拟脚本
            cmd = [
                "python", 
                "mock_inference.py",
                "--config", str(config_file),
                "--output_dir", str(output_dir)
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                cwd=str(FACECHAIN_DIR),
                capture_output=True,
                text=True
            )
            
            logger.info(f"模拟生成完成: {result.stdout}")
            return True
        except Exception as e:
            logger.error(f"模拟生成失败: {e}")
            return False
    
    def _prepare_config(self, task_id, photos, styles):
        """准备FaceChain的配置文件"""
        # 确保照片列表中的路径都存在
        valid_photos = []
        for photo in photos:
            if os.path.exists(photo):
                valid_photos.append(photo)
            else:
                logger.warning(f"照片路径不存在: {photo}")
        
        if not valid_photos:
            logger.error("没有有效的照片路径")
            raise ValueError("没有有效的照片路径")
        
        # 确定风格
        style = styles[0] if styles else "portrait"
        # 映射风格名称（确保适配FaceChain支持的风格）
        style_mapping = {
            "婚纱": "wedding",
            "油画": "oil",
            "古风": "ancient",
            "漫画": "comic",
            "portrait": "portrait"
        }
        mapped_style = style_mapping.get(style, "portrait")
        
        # 创建配置
        config = {
            "task_id": task_id,
            "use_pose_model": False,  # 不使用姿势控制
            "input_img_paths": valid_photos,
            "num_generate": 5,  # 生成5张图片
            "multiplier_style": 0.25,  # 风格权重
            "style": mapped_style,  # 使用映射后的风格
        }
        
        # 保存到临时文件
        fd, path = tempfile.mkstemp(suffix='.json', prefix=f'task_{task_id}_')
        with os.fdopen(fd, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"已创建配置文件: {path}, 风格: {mapped_style}")
        return path

# 单例模式
facechain_generator = FaceChainGenerator() 