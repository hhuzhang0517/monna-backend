import os
import cv2
import uuid
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
from app.core.config import settings

# 移除mediapipe依赖
# 模拟实现

class ImageUtils:
    @staticmethod
    def read_image(file_path: str) -> np.ndarray:
        """读取图像文件"""
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"无法读取图像: {file_path}")
        return img
    
    @staticmethod
    def save_image(img: np.ndarray, file_path: str) -> str:
        """保存图像到文件"""
        success = cv2.imwrite(file_path, img)
        if not success:
            raise ValueError(f"无法保存图像到: {file_path}")
        return file_path
    
    @staticmethod
    def resize_image(img: np.ndarray, max_size: int = 1024) -> np.ndarray:
        """调整图像大小，保持纵横比"""
        height, width = img.shape[:2]
        
        # 如果图像已经小于最大尺寸，则不调整
        if max(height, width) <= max_size:
            return img
        
        # 计算调整比例
        ratio = max_size / max(height, width)
        new_height = int(height * ratio)
        new_width = int(width * ratio)
        
        # 调整图像大小
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_img
    
    @staticmethod
    def detect_face(img: np.ndarray) -> Dict[str, Any]:
        """模拟人脸检测，返回边界框和关键点"""
        # 简单的模拟实现，假设图像中央有一个人脸
        h, w = img.shape[:2]
        
        # 估计人脸在中心位置，大小为图像的40%
        face_width = int(w * 0.4)
        face_height = int(h * 0.4)
        xmin = (w - face_width) // 2
        ymin = (h - face_height) // 2
        
        # 模拟5个关键点（眼睛、鼻子、嘴巴两角）
        keypoints = {
            "kp0": (xmin + face_width // 3, ymin + face_height // 3),  # 左眼
            "kp1": (xmin + 2 * face_width // 3, ymin + face_height // 3),  # 右眼
            "kp2": (xmin + face_width // 2, ymin + face_height // 2),  # 鼻子
            "kp3": (xmin + face_width // 3, ymin + 2 * face_height // 3),  # 嘴巴左角
            "kp4": (xmin + 2 * face_width // 3, ymin + 2 * face_height // 3),  # 嘴巴右角
        }
        
        return {
            "detected": True,
            "bbox": (xmin, ymin, face_width, face_height),
            "keypoints": keypoints
        }
    
    @staticmethod
    def align_face(img: np.ndarray, face_data: Dict[str, Any], output_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
        """根据关键点对齐人脸"""
        if not face_data["detected"]:
            raise ValueError("未检测到人脸，无法对齐")
        
        # 提取边界框
        xmin, ymin, width, height = face_data["bbox"]
        
        # 计算中心点和缩放比例
        center = (xmin + width // 2, ymin + height // 2)
        
        # 增加边界框大小以包含更多上下文
        scale_factor = 1.5
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # 计算新的边界框
        new_xmin = max(0, center[0] - new_width // 2)
        new_ymin = max(0, center[1] - new_height // 2)
        new_xmax = min(img.shape[1], new_xmin + new_width)
        new_ymax = min(img.shape[0], new_ymin + new_height)
        
        # 裁剪人脸区域
        face_img = img[new_ymin:new_ymax, new_xmin:new_xmax]
        
        # 调整大小到目标尺寸
        aligned_face = cv2.resize(face_img, output_size, interpolation=cv2.INTER_AREA)
        
        return aligned_face
    
    @staticmethod
    def extract_face_mesh(img: np.ndarray) -> Dict[str, Any]:
        """模拟提取人脸网格关键点"""
        # 简单模拟实现，生成网格点
        h, w = img.shape[:2]
        
        # 估计人脸在中心位置
        face_width = int(w * 0.4)
        face_height = int(h * 0.4)
        xmin = (w - face_width) // 2
        ymin = (h - face_height) // 2
        
        # 创建一个简单的网格点集合(模拟468个点)
        landmarks = {}
        
        # 生成简单的面部网格(5x5)
        for i in range(25):
            row = i // 5
            col = i % 5
            
            x = xmin + col * face_width // 4
            y = ymin + row * face_height // 4
            z = 0  # 简化为平面
            
            landmarks[i] = (x, y, z)
        
        return {
            "detected": True,
            "landmarks": landmarks
        }
    
    @staticmethod
    def generate_uuid_filename(original_filename: str, directory: Path) -> str:
        """生成唯一文件名"""
        extension = os.path.splitext(original_filename)[1]
        new_filename = f"{uuid.uuid4()}{extension}"
        return os.path.join(directory, new_filename)
    
    @staticmethod
    def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
        """将BGR格式转换为RGB格式"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
        """将RGB格式转换为BGR格式"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)