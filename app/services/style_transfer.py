import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from app.utils.image_utils import ImageUtils
from app.services.face_features import FaceFeatureExtractor, FaceFeatureLevel
from app.core.config import settings

# 配置日志
logger = logging.getLogger(__name__)

class StyleTransferService:
    """照片风格转换服务，用于将用户照片转换为特定风格"""
    
    def __init__(self):
        """初始化风格转换服务"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_utils = ImageUtils()
        self.face_extractor = FaceFeatureExtractor()
        self.wedding_model = None
        self.model_style_model = None
        logger.info(f"StyleTransferService初始化，设备: {self.device}")
    
    def load_model(self, style="wedding"):
        """加载对应风格的模型"""
        logger.info(f"加载{style}风格模型")
        
        if style == "wedding" and self.wedding_model is None:
            # 在实际项目中，这里应该加载预训练模型
            # 例如: self.wedding_model = WeddingStyleModel().to(self.device)
            # self.wedding_model.load_state_dict(torch.load(settings.MODEL_PATHS.get("wedding")))
            # self.wedding_model.eval()
            logger.info("婚纱照风格模型加载成功")
        
        elif style == "model" and self.model_style_model is None:
            # 在实际项目中，这里应该加载预训练模型
            # 例如: self.model_style_model = ModelStyleModel().to(self.device)
            # self.model_style_model.load_state_dict(torch.load(settings.MODEL_PATHS.get("model")))
            # self.model_style_model.eval()
            logger.info("男模风风格模型加载成功")
    
    def transfer_style(self, image_paths: List[str], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        将多张照片转换为特定风格
        
        Args:
            image_paths: 原始图像路径列表
            options: 选项字典，包括style(风格类型)和gender(性别)等
        
        Returns:
            包含结果路径的字典
        """
        start_time = time.time()
        style = options.get("style", "wedding")
        gender = options.get("gender", "auto")
        
        logger.info(f"开始风格转换: 风格={style}, 性别={gender}, 图片数量={len(image_paths)}")
        
        # 加载相应的模型
        self.load_model(style)
        
        # 收集所有人脸特征，用于分析用户的面部特征
        face_features = []
        for image_path in image_paths:
            logger.debug(f"处理图像: {image_path}")
            img = self.image_utils.read_image(image_path)
            
            # 提取人脸特征
            face_result = self.face_extractor.extract_face_features(img, FaceFeatureLevel.DETAILED)
            if face_result["face_detected"]:
                face_features.extend(face_result["faces"])
                logger.debug(f"在图像中检测到{len(face_result['faces'])}张人脸")
        
        # 如果性别未指定，尝试自动检测
        if gender == "auto" and face_features:
            # 统计检测到的性别
            gender_counts = {"male": 0, "female": 0}
            for face in face_features:
                if "attributes" in face and "gender" in face["attributes"]:
                    detected_gender = face["attributes"]["gender"]
                    if detected_gender == "male":
                        gender_counts["male"] += 1
                    elif detected_gender == "female":
                        gender_counts["female"] += 1
            
            # 确定主要性别
            if gender_counts["male"] > gender_counts["female"]:
                gender = "male"
            else:
                gender = "female"
            
            logger.info(f"自动检测到性别: {gender} (男:{gender_counts['male']}, 女:{gender_counts['female']})")
        
        # 创建结果数组
        result_images = []
        
        # 执行风格转换
        if style == "wedding":
            result_images = self._wedding_style_transfer(image_paths, face_features, gender, options)
        elif style == "model":
            result_images = self._model_style_transfer(image_paths, face_features, gender, options)
        else:
            # 默认简单处理
            result_images = self._simple_style_transfer(image_paths, style, options)
        
        # 计算处理用时
        process_time = time.time() - start_time
        logger.info(f"风格转换完成，耗时: {process_time:.2f}秒，生成{len(result_images)}张结果图像")
        
        return {
            "result_paths": result_images,
            "style": style,
            "gender": gender,
            "process_time": process_time
        }
    
    def _wedding_style_transfer(self, image_paths: List[str], face_features: List[Dict], gender: str, options: Dict[str, Any]) -> List[str]:
        """
        婚纱照风格转换
        
        Args:
            image_paths: 图像路径列表
            face_features: 人脸特征列表
            gender: 性别
            options: 其他选项
        
        Returns:
            结果图像路径列表
        """
        logger.info(f"执行婚纱照风格转换，图片数量: {len(image_paths)}")
        
        # 这里应该使用实际的婚纱照风格转换模型
        # 目前简单模拟处理过程
        result_paths = []
        
        for idx, image_path in enumerate(image_paths):
            # 读取图像
            img = self.image_utils.read_image(image_path)
            
            # 简单模拟风格化效果（实际项目中应该使用深度学习模型）
            result_img = self._simple_wedding_effect(img, gender)
            
            # 保存结果
            output_path = str(Path(image_path).with_suffix(f'.wedding_result_{idx}.jpg'))
            self.image_utils.save_image(result_img, output_path)
            result_paths.append(output_path)
            
            logger.debug(f"生成婚纱照风格图像: {output_path}")
        
        # 额外生成几张组合图像
        if len(image_paths) >= 2:
            for i in range(3):  # 生成3张额外组合图像
                # 随机选择两张图片组合
                import random
                img1_path = random.choice(image_paths)
                img2_path = random.choice(image_paths)
                img1 = self.image_utils.read_image(img1_path)
                img2 = self.image_utils.read_image(img2_path)
                
                # 简单拼接（实际应该有更高级的组合处理）
                combined_img = self._combine_images(img1, img2)
                
                # 保存结果
                output_path = str(Path(image_paths[0]).parent / f"wedding_combined_{i}.jpg")
                self.image_utils.save_image(combined_img, output_path)
                result_paths.append(output_path)
                
                logger.debug(f"生成婚纱照组合图像: {output_path}")
        
        return result_paths
    
    def _model_style_transfer(self, image_paths: List[str], face_features: List[Dict], gender: str, options: Dict[str, Any]) -> List[str]:
        """
        男模风风格转换
        
        Args:
            image_paths: 图像路径列表
            face_features: 人脸特征列表
            gender: 性别
            options: 其他选项
        
        Returns:
            结果图像路径列表
        """
        logger.info(f"执行男模风风格转换，图片数量: {len(image_paths)}")
        
        # 这里应该使用实际的男模风风格转换模型
        # 目前简单模拟处理过程
        result_paths = []
        
        for idx, image_path in enumerate(image_paths):
            # 读取图像
            img = self.image_utils.read_image(image_path)
            
            # 简单模拟风格化效果（实际项目中应该使用深度学习模型）
            result_img = self._simple_model_effect(img)
            
            # 保存结果
            output_path = str(Path(image_path).with_suffix(f'.model_result_{idx}.jpg'))
            self.image_utils.save_image(result_img, output_path)
            result_paths.append(output_path)
            
            logger.debug(f"生成男模风风格图像: {output_path}")
        
        return result_paths
    
    def _simple_style_transfer(self, image_paths: List[str], style: str, options: Dict[str, Any]) -> List[str]:
        """
        简单风格转换（用于未实现的风格类型）
        
        Args:
            image_paths: 图像路径列表
            style: 风格类型
            options: 其他选项
        
        Returns:
            结果图像路径列表
        """
        logger.info(f"执行简单风格转换: {style}，图片数量: {len(image_paths)}")
        
        result_paths = []
        
        for idx, image_path in enumerate(image_paths):
            # 读取图像
            img = self.image_utils.read_image(image_path)
            
            # 简单滤镜效果
            if style == "oil-painting":
                result_img = self._simple_oil_painting_effect(img)
            else:
                # 默认简单增强
                result_img = self._simple_enhance(img)
            
            # 保存结果
            output_path = str(Path(image_path).with_suffix(f'.{style}_result_{idx}.jpg'))
            self.image_utils.save_image(result_img, output_path)
            result_paths.append(output_path)
            
            logger.debug(f"生成简单风格图像: {output_path}")
        
        return result_paths
    
    def _simple_wedding_effect(self, img: np.ndarray, gender: str) -> np.ndarray:
        """简单的婚纱照效果（仅用于演示）"""
        # 增加亮度和对比度
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        bright_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        # 柔化皮肤
        blurred = cv2.GaussianBlur(bright_img, (0, 0), 10)
        result = cv2.addWeighted(bright_img, 0.7, blurred, 0.3, 0)
        
        # 增加温暖色调
        result = self._add_warm_tone(result)
        
        return result
    
    def _simple_model_effect(self, img: np.ndarray) -> np.ndarray:
        """简单的男模风效果（仅用于演示）"""
        # 转为黑白
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # 增加对比度
        alpha = 1.5  # 对比度控制
        beta = 0     # 亮度控制
        contrast_img = cv2.convertScaleAbs(gray_3channel, alpha=alpha, beta=beta)
        
        # 锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(contrast_img, -1, kernel)
        
        return result
    
    def _simple_oil_painting_effect(self, img: np.ndarray) -> np.ndarray:
        """简单的油画效果（仅用于演示）"""
        # 使用stylization滤镜
        result = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
        return result
    
    def _simple_enhance(self, img: np.ndarray) -> np.ndarray:
        """简单的图像增强（仅用于演示）"""
        # 自动白平衡
        result = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
        return result
    
    def _add_warm_tone(self, img: np.ndarray) -> np.ndarray:
        """添加温暖色调"""
        # 分解为BGR通道
        b, g, r = cv2.split(img)
        
        # 增加红色和绿色，减少蓝色
        r = cv2.add(r, 10)
        g = cv2.add(g, 5)
        b = cv2.subtract(b, 5)
        
        # 合并通道
        return cv2.merge([b, g, r])
    
    def _combine_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """简单的图像组合（仅用于演示）"""
        # 调整尺寸
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # 计算组合图像尺寸
        h_combined = max(h1, h2)
        w_combined = w1 + w2
        
        # 创建组合图像
        combined = np.zeros((h_combined, w_combined, 3), dtype=np.uint8)
        
        # 调整图像大小
        img1_resized = cv2.resize(img1, (w1, h_combined))
        img2_resized = cv2.resize(img2, (w2, h_combined))
        
        # 放置两张图片
        combined[:, :w1] = img1_resized
        combined[:, w1:] = img2_resized
        
        return combined 