import os
import sys
import json
import logging
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path

from app.core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

# 确保日志目录存在
logs_dir = settings.BASE_DIR / "logs"
os.makedirs(logs_dir, exist_ok=True)

class FaceChainService:
    """FaceChain AI人像生成服务，提供与FaceChain模型的集成"""
    
    def __init__(self):
        """初始化FaceChain服务"""
        self.facechain_dir = settings.BASE_DIR / "models" / "facechain"
        self.python_executable = sys.executable
        
        # 确保目录存在
        os.makedirs(settings.UPLOAD_DIR / "facechain", exist_ok=True)
        os.makedirs(settings.RESULTS_DIR / "facechain", exist_ok=True)
        
        logger.info(f"FaceChainService初始化完成，FaceChain目录: {self.facechain_dir}")
    
    def generate_portrait(
        self,
        input_img_path: str,
        output_dir: str,
        style_name: str,
        num_generate: int = 4,
        multiplier_style: float = 0.25,
        use_pose: bool = False,
        pose_image_path: Optional[str] = None,
        use_face_swap: bool = True
    ) -> List[str]:
        """
        生成AI人像
        
        Args:
            input_img_path: 输入图像的路径
            output_dir: 输出目录
            style_name: 风格名称
            num_generate: 生成图像的数量 
            multiplier_style: 风格强度
            use_pose: 是否使用姿势引导
            pose_image_path: 姿势引导图像的路径
            use_face_swap: 是否使用面部交换
            
        Returns:
            List[str]: 生成的图像文件路径列表
        """
        logger.info(f"开始生成AI人像，风格: {style_name}, 数量: {num_generate}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 构建Python代码作为字符串
            python_code = f"""
import os
import sys
import json

# 添加facechain目录到Python路径
sys.path.append('{str(self.facechain_dir)}')

# 导入facechain模块
try:
    from facechain.inference_face import GenPortrait
    from facechain.utils import snapshot_download
    from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, base_models
    import cv2
except ImportError as e:
    print(f"错误: 无法导入FaceChain模块: {{e}}")
    sys.exit(1)

# 尝试加载风格配置
try:
    styles = []
    for base_model in base_models:
        style_in_base = []
        style_folder = f"styles/{{base_model['name']}}"
        if not os.path.exists(style_folder):
            print(f"警告: 风格目录不存在: {{style_folder}}")
            continue
            
        files = os.listdir(style_folder)
        files.sort()
        for file in files:
            file_path = os.path.join(style_folder, file)
            if not file.lower().endswith('.json'):
                continue
                
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    data = json.load(f)
                    style_in_base.append(data['name'])
                    styles.append(data)
            except Exception as e:
                print(f"警告: 无法读取风格配置文件 {{file_path}}: {{e}}")
                continue
                
        base_model['style_list'] = style_in_base

    # 查找指定的风格
    target_style = None
    for s in styles:
        if s['name'] == '{style_name}':
            target_style = s
            break

    if not target_style:
        print(f"错误: 未找到指定的风格 '{style_name}'")
        print(f"可用的风格有: {{[s['name'] for s in styles]}}")
        sys.exit(1)
        
    # 设置参数
    use_pose = {str(use_pose).lower()}
    input_img_path = '{input_img_path}'
    pose_image = '{pose_image_path if use_pose and pose_image_path else "None"}'
    pose_image = pose_image if pose_image != 'None' else None
    num_generate = {num_generate}
    multiplier_style = {multiplier_style}
    output_dir = '{output_dir}'
    use_face_swap = {str(use_face_swap).lower()}
    base_model_idx = 0  # 默认使用第一个基础模型
    
    if hasattr(target_style, 'model_id'):
        model_id = target_style.get('model_id')
    else:
        model_id = None
        print(f"警告: 风格配置中没有model_id字段，将使用默认模型")

    # 获取风格模型路径
    if model_id is None:
        style_model_path = None
        pos_prompt = pos_prompt_with_cloth.format(target_style.get('add_prompt_style', ''))
    else:
        if os.path.exists(model_id):
            model_dir = model_id
        else:
            model_dir = snapshot_download(model_id, revision=target_style.get('revision', 'main'))
        style_model_path = os.path.join(model_dir, target_style.get('bin_file', 'pytorch_model.bin'))
        pos_prompt = pos_prompt_with_style.format(target_style.get('add_prompt_style', ''))

    # 准备姿势图像
    if not use_pose:
        pose_image = None

    # 初始化生成器并执行生成
    gen_portrait = GenPortrait()
    outputs = gen_portrait(
        num_generate=num_generate, 
        model_id=base_model_idx, 
        style_model_path=style_model_path, 
        pose_img=pose_image,
        ref_img=input_img_path,
        pos_prompt=pos_prompt, 
        neg_prompt=neg_prompt, 
        strength=multiplier_style,
        use_face_swap=use_face_swap
    )

    # 保存生成的图像
    output_files = []
    for i, out_tmp in enumerate(outputs):
        output_path = os.path.join(output_dir, f'result_{{i}}.png')
        cv2.imwrite(output_path, out_tmp)
        output_files.append(output_path)

    # 输出结果文件路径列表，用于后续处理
    print(json.dumps(output_files))
    
except Exception as e:
    import traceback
    print(f"错误: {{e}}")
    print(traceback.format_exc())
    sys.exit(1)
"""
            
            # 将代码写入临时文件
            temp_script_path = os.path.join(output_dir, "generate_portrait.py")
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(python_code)
            
            # 构建命令
            cmd = [self.python_executable, temp_script_path]
            
            # 将当前目录切换到facechain目录
            current_dir = os.getcwd()
            os.chdir(str(self.facechain_dir))
            
            # 执行命令
            logger.info(f"执行FaceChain推理命令...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待处理完成
            stdout, stderr = process.communicate()
            
            # 恢复当前目录
            os.chdir(current_dir)
            
            if process.returncode != 0:
                logger.error(f"FaceChain处理失败: {stderr}")
                return []
            
            # 解析输出结果
            try:
                result_files = json.loads(stdout.strip())
                logger.info(f"FaceChain生成了 {len(result_files)} 个结果文件")
                return result_files
                
            except json.JSONDecodeError:
                logger.error(f"无法解析FaceChain输出: {stdout}")
                return []
                
        except Exception as e:
            logger.exception(f"生成AI人像时出错: {e}")
            return []
    
    def generate_portrait_inpaint(
        self,
        input_img_path: str,
        template_img_path: str,
        output_dir: str,
        num_faces: int = 1,
        selected_face: int = 1,
        use_face_swap: bool = True
    ) -> List[str]:
        """
        使用FaceChain的修复功能生成AI人像
        
        Args:
            input_img_path: 输入图像的路径
            template_img_path: 模板图像的路径
            output_dir: 输出目录
            num_faces: 检测的人脸数量
            selected_face: 选择的人脸序号
            use_face_swap: 是否使用面部交换
            
        Returns:
            List[str]: 生成的图像文件路径列表
        """
        logger.info(f"开始生成修复版AI人像，模板: {template_img_path}")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 构建Python代码作为字符串
            python_code = f"""
import os
import sys
import json

# 添加facechain目录到Python路径
sys.path.append('{str(self.facechain_dir)}')

# 导入facechain模块
try:
    from facechain.inference_inpaint import GenPortraitInpaint
    import cv2
except ImportError as e:
    print(f"错误: 无法导入FaceChain模块: {{e}}")
    sys.exit(1)

try:
    # 设置参数
    input_img_path = '{input_img_path}'
    template_img_path = '{template_img_path}'
    output_dir = '{output_dir}'
    num_faces = {num_faces}
    selected_face = {selected_face - 1}  # 转为0-indexed
    use_face_swap = {str(use_face_swap).lower()}

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化生成器并执行生成
    gen_inpaint = GenPortraitInpaint()
    outputs = gen_inpaint(
        template_img=template_img_path,
        ref_img=input_img_path,
        num_faces=num_faces,
        selected_face=selected_face,
        use_face_swap=use_face_swap
    )

    # 保存生成的图像
    output_files = []
    for i, out_tmp in enumerate(outputs):
        output_path = os.path.join(output_dir, f'inpaint_result_{{i}}.png')
        cv2.imwrite(output_path, out_tmp)
        output_files.append(output_path)

    # 输出结果文件路径列表，用于后续处理
    print(json.dumps(output_files))
    
except Exception as e:
    import traceback
    print(f"错误: {{e}}")
    print(traceback.format_exc())
    sys.exit(1)
"""
            
            # 将代码写入临时文件
            temp_script_path = os.path.join(output_dir, "generate_inpaint.py")
            with open(temp_script_path, "w", encoding="utf-8") as f:
                f.write(python_code)
            
            # 构建命令
            cmd = [self.python_executable, temp_script_path]
            
            # 将当前目录切换到facechain目录
            current_dir = os.getcwd()
            os.chdir(str(self.facechain_dir))
            
            # 执行命令
            logger.info(f"执行FaceChain修复命令...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待处理完成
            stdout, stderr = process.communicate()
            
            # 恢复当前目录
            os.chdir(current_dir)
            
            if process.returncode != 0:
                logger.error(f"FaceChain修复处理失败: {stderr}")
                return []
            
            # 解析输出结果
            try:
                result_files = json.loads(stdout.strip())
                logger.info(f"FaceChain生成了 {len(result_files)} 个修复结果文件")
                return result_files
                
            except json.JSONDecodeError:
                logger.error(f"无法解析FaceChain修复输出: {stdout}")
                return []
                
        except Exception as e:
            logger.exception(f"生成修复版AI人像时出错: {e}")
            return []
