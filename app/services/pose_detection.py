import cv2
import numpy as np
import time
import mediapipe as mp
from enum import Enum
from typing import Dict, Any, List, Tuple, Optional, Union
import uuid
from pathlib import Path
import os
import json
from functools import lru_cache

# 导入特征提取模块
from app.services.face_features import FaceFeatureExtractor, FaceFeatureLevel

# 初始化MediaPipe工具
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class ImageType(str, Enum):
    """图像类型枚举"""
    UNKNOWN = "unknown"
    PORTRAIT = "portrait"      # 人像
    FULL_BODY = "full_body"    # 全身像
    GROUP = "group"            # 人群
    PRODUCT = "product"        # 商品
    LANDSCAPE = "landscape"    # 风景
    TEXT = "text"              # 文字

class PoseEstimationLevel(str, Enum):
    """姿态估计精度级别"""
    FAST = "fast"          # 快速模式(低精度)
    STANDARD = "standard"  # 标准模式(中等精度)
    ACCURATE = "accurate"  # 精确模式(高精度)

class BodyPart(str, Enum):
    """身体部位枚举"""
    NOSE = "nose"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"

# MediaPipe姿态关键点映射
MP_POSE_LANDMARKS = {
    0: BodyPart.NOSE,
    1: BodyPart.LEFT_EYE,
    2: BodyPart.RIGHT_EYE,
    3: BodyPart.LEFT_EAR,
    4: BodyPart.RIGHT_EAR,
    5: BodyPart.LEFT_SHOULDER,
    6: BodyPart.RIGHT_SHOULDER,
    7: BodyPart.LEFT_ELBOW,
    8: BodyPart.RIGHT_ELBOW,
    9: BodyPart.LEFT_WRIST,
    10: BodyPart.RIGHT_WRIST,
    11: BodyPart.LEFT_HIP,
    12: BodyPart.RIGHT_HIP,
    13: BodyPart.LEFT_KNEE,
    14: BodyPart.RIGHT_KNEE,
    15: BodyPart.LEFT_ANKLE,
    16: BodyPart.RIGHT_ANKLE
}

class PoseDetectionLevel(str, Enum):
    """人体姿态检测精度级别"""
    FAST = "fast"          # 快速模式(低精度)
    STANDARD = "standard"  # 标准模式(中等精度)
    ACCURATE = "accurate"  # 精确模式(高精度)

class PoseType(str, Enum):
    """姿态类型"""
    STANDING = "standing"      # 站立
    SITTING = "sitting"        # 坐姿
    LYING = "lying"            # 躺卧
    WALKING = "walking"        # 行走
    RUNNING = "running"        # 跑步
    DYNAMIC = "dynamic"        # 其他运动
    UNKNOWN = "unknown"        # 未知

class PoseEstimator:
    """人体姿态估计类"""
    
    def __init__(self):
        """初始化姿态估计器"""
        self._pose_model = None
        self._model_complexity = {
            PoseEstimationLevel.FAST: 0,
            PoseEstimationLevel.STANDARD: 1,
            PoseEstimationLevel.ACCURATE: 2
        }
    
    def detect_pose(self, img: np.ndarray, 
                    level: PoseEstimationLevel = PoseEstimationLevel.STANDARD,
                    min_detection_confidence: float = 0.5) -> Dict[str, Any]:
        """
        检测图像中的人体姿态
        
        Args:
            img: 输入图像(BGR格式)
            level: 姿态估计精度级别
            min_detection_confidence: 最小检测置信度
            
        Returns:
            姿态检测结果字典
        """
        start_time = time.time()
        
        # 初始化结果字典
        result = {
            "pose_detected": False,
            "estimation_level": level,
            "process_time": 0
        }
        
        # 转换为RGB (MediaPipe需要RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 选择模型复杂度
        model_complexity = self._model_complexity.get(level, 1)
        
        # 使用MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=True,  # 启用分割
            min_detection_confidence=min_detection_confidence
        ) as pose_model:
            # 处理图像
            results = pose_model.process(rgb_img)
            
            if not results.pose_landmarks:
                result["process_time"] = time.time() - start_time
                return result
            
            # 提取姿态关键点
            landmarks = []
            landmark_dict = {}
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # 转换为像素坐标
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                
                # 标准化为相对坐标 (0-1)
                normalized_x = landmark.x
                normalized_y = landmark.y
                normalized_z = landmark.z
                
                # 将关键点添加到列表
                landmarks.append({
                    "idx": idx,
                    "name": MP_POSE_LANDMARKS.get(idx, f"point_{idx}").value,
                    "x": normalized_x,
                    "y": normalized_y,
                    "z": normalized_z,
                    "visibility": landmark.visibility,
                    "px": px,
                    "py": py
                })
                
                # 同时以身体部位名称为键存储
                if idx in MP_POSE_LANDMARKS:
                    part_name = MP_POSE_LANDMARKS[idx].value
                    landmark_dict[part_name] = {
                        "x": normalized_x,
                        "y": normalized_y,
                        "z": normalized_z,
                        "visibility": landmark.visibility,
                        "px": px,
                        "py": py
                    }
            
            # 计算人体边界框
            pose_bbox = self._compute_pose_bbox(landmarks, h, w)
            
            # 分析姿态
            pose_analysis = self._analyze_pose(landmark_dict)
            
            # 如果启用了分割，处理分割掩码
            if results.segmentation_mask is not None:
                mask = results.segmentation_mask
                condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                
                # 创建分割图像
                bg_color = (192, 192, 192)  # 灰色背景
                fg_image = np.zeros(img.shape, dtype=np.uint8)
                fg_image[:] = bg_color
                fg_image = np.where(condition, img, fg_image)
                
                # 保存分割掩码
                mask_image = (mask * 255).astype(np.uint8)
                mask_path = str(Path(str(uuid.uuid4())).with_suffix('.pose_mask.png'))
                cv2.imwrite(mask_path, mask_image)
                
                # 保存分割图像
                segmented_path = str(Path(str(uuid.uuid4())).with_suffix('.pose_segmented.jpg'))
                cv2.imwrite(segmented_path, fg_image)
                
                result.update({
                    "segmentation_mask_path": mask_path,
                    "segmented_image_path": segmented_path
                })
            
            # 生成姿态可视化图像
            annotated_image = self._visualize_pose(img.copy(), results)
            annotated_path = str(Path(str(uuid.uuid4())).with_suffix('.pose_visualization.jpg'))
            cv2.imwrite(annotated_path, annotated_image)
            
            # 更新结果
            result.update({
                "pose_detected": True,
                "landmarks": landmarks,
                "landmark_dict": landmark_dict,
                "pose_bbox": pose_bbox,
                "visualization_path": annotated_path,
                "analysis": pose_analysis
            })
            
            # 记录处理时间
            result["process_time"] = time.time() - start_time
            
            return result
    
    def _compute_pose_bbox(self, landmarks: List[Dict[str, Any]], img_height: int, img_width: int) -> Dict[str, Any]:
        """
        计算姿态边界框
        
        Args:
            landmarks: 姿态关键点列表
            img_height: 图像高度
            img_width: 图像宽度
            
        Returns:
            边界框字典 (x, y, width, height, 以及相对坐标)
        """
        # 初始化边界值
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0
        
        # 遍历所有关键点找到边界
        valid_landmarks = 0
        
        for lm in landmarks:
            # 仅考虑可见性较高的点
            if lm["visibility"] > 0.5:
                x, y = lm["px"], lm["py"]
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
                valid_landmarks += 1
        
        # 如果没有找到有效的关键点，返回整个图像
        if valid_landmarks == 0 or x_min == float('inf'):
            x_min, y_min, width, height = 0, 0, img_width, img_height
        else:
            # 确保边界在图像内
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            width = min(img_width - x_min, x_max - x_min)
            height = min(img_height - y_min, y_max - y_min)
            
            # 扩大边界框以包含整个人体
            padding = 0.1  # 边界框扩展比例
            x_min = max(0, int(x_min - width * padding))
            y_min = max(0, int(y_min - height * padding))
            width = min(img_width - x_min, int(width * (1 + 2 * padding)))
            height = min(img_height - y_min, int(height * (1 + 2 * padding)))
        
        # 计算相对坐标
        x_rel = x_min / img_width
        y_rel = y_min / img_height
        width_rel = width / img_width
        height_rel = height / img_height
        
        return {
            "x": x_min,
            "y": y_min,
            "width": width,
            "height": height,
            "x_rel": x_rel,
            "y_rel": y_rel,
            "width_rel": width_rel,
            "height_rel": height_rel
        }
    
    def _visualize_pose(self, img: np.ndarray, pose_results) -> np.ndarray:
        """
        可视化姿态检测结果
        
        Args:
            img: 输入图像
            pose_results: MediaPipe姿态检测结果
            
        Returns:
            标注后的图像
        """
        # 绘制姿态关键点和连接线
        mp_drawing.draw_landmarks(
            img,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        return img
    
    def _analyze_pose(self, landmark_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析人体姿态的特征
        
        Args:
            landmark_dict: 关键点字典，以身体部位为键
            
        Returns:
            姿态分析结果
        """
        analysis = {}
        
        # 检查是否有足够的关键点进行分析
        if len(landmark_dict) < 10:
            analysis["pose_quality"] = "insufficient_keypoints"
            return analysis
        
        # 分析手臂状态
        left_arm_visible = all(k in landmark_dict for k in 
                              [BodyPart.LEFT_SHOULDER.value, BodyPart.LEFT_ELBOW.value, BodyPart.LEFT_WRIST.value])
        right_arm_visible = all(k in landmark_dict for k in 
                               [BodyPart.RIGHT_SHOULDER.value, BodyPart.RIGHT_ELBOW.value, BodyPart.RIGHT_WRIST.value])
        
        # 分析腿部状态
        left_leg_visible = all(k in landmark_dict for k in 
                              [BodyPart.LEFT_HIP.value, BodyPart.LEFT_KNEE.value, BodyPart.LEFT_ANKLE.value])
        right_leg_visible = all(k in landmark_dict for k in 
                               [BodyPart.RIGHT_HIP.value, BodyPart.RIGHT_KNEE.value, BodyPart.RIGHT_ANKLE.value])
        
        # 姿态质量
        visible_parts = sum(lm["visibility"] > 0.5 for lm in landmark_dict.values())
        total_parts = len(MP_POSE_LANDMARKS)
        pose_quality = visible_parts / total_parts
        
        analysis.update({
            "pose_quality": pose_quality,
            "arms_visible": {
                "left": left_arm_visible,
                "right": right_arm_visible
            },
            "legs_visible": {
                "left": left_leg_visible,
                "right": right_leg_visible
            }
        })
        
        # 分析姿势类型(站立、坐姿、侧身等)
        pose_type = self._determine_pose_type(landmark_dict)
        if pose_type:
            analysis["pose_type"] = pose_type
        
        # 计算身体各部分角度
        angles = self._calculate_body_angles(landmark_dict)
        if angles:
            analysis["angles"] = angles
        
        return analysis
    
    def _determine_pose_type(self, landmark_dict: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """
        确定姿势类型
        
        Args:
            landmark_dict: 关键点字典
            
        Returns:
            姿势类型字符串
        """
        # 检查关键点可见性
        if not all(k in landmark_dict for k in [
            BodyPart.LEFT_HIP.value, BodyPart.RIGHT_HIP.value,
            BodyPart.LEFT_SHOULDER.value, BodyPart.RIGHT_SHOULDER.value,
            BodyPart.LEFT_ANKLE.value, BodyPart.RIGHT_ANKLE.value
        ]):
            return None
        
        # 获取关键身体部位
        left_hip = landmark_dict[BodyPart.LEFT_HIP.value]
        right_hip = landmark_dict[BodyPart.RIGHT_HIP.value]
        left_shoulder = landmark_dict[BodyPart.LEFT_SHOULDER.value]
        right_shoulder = landmark_dict[BodyPart.RIGHT_SHOULDER.value]
        left_ankle = landmark_dict[BodyPart.LEFT_ANKLE.value]
        right_ankle = landmark_dict[BodyPart.RIGHT_ANKLE.value]
        
        # 计算躯干高度(肩膀到臀部的距离)
        torso_height = ((left_shoulder["y"] + right_shoulder["y"]) / 2 - 
                        (left_hip["y"] + right_hip["y"]) / 2)
        
        # 计算腿部高度(臀部到脚踝的距离)
        leg_height = ((left_hip["y"] + right_hip["y"]) / 2 - 
                      (left_ankle["y"] + right_ankle["y"]) / 2)
        
        # 躯干垂直度(肩膀和臀部水平对齐程度)
        shoulder_alignment = abs(left_shoulder["y"] - right_shoulder["y"])
        hip_alignment = abs(left_hip["y"] - right_hip["y"])
        vertical_alignment = (shoulder_alignment + hip_alignment) / 2
        
        # 根据这些特征确定姿势类型
        if abs(torso_height) < 0.1:  # 躯干接近水平
            return "lying"
        elif leg_height < 0.1:  # 腿部高度很小
            return "sitting"
        elif vertical_alignment > 0.2:  # 肩膀和臀部不对齐
            return "leaning"
        else:
            return "standing"
    
    def _calculate_body_angles(self, landmark_dict: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        计算身体各部分角度
        
        Args:
            landmark_dict: 关键点字典
            
        Returns:
            身体角度字典
        """
        angles = {}
        
        # 计算肘部角度
        if all(k in landmark_dict for k in [
            BodyPart.LEFT_SHOULDER.value, BodyPart.LEFT_ELBOW.value, BodyPart.LEFT_WRIST.value
        ]):
            left_shoulder = landmark_dict[BodyPart.LEFT_SHOULDER.value]
            left_elbow = landmark_dict[BodyPart.LEFT_ELBOW.value]
            left_wrist = landmark_dict[BodyPart.LEFT_WRIST.value]
            
            angles["left_elbow"] = self._calculate_angle(
                (left_shoulder["x"], left_shoulder["y"]),
                (left_elbow["x"], left_elbow["y"]),
                (left_wrist["x"], left_wrist["y"])
            )
        
        if all(k in landmark_dict for k in [
            BodyPart.RIGHT_SHOULDER.value, BodyPart.RIGHT_ELBOW.value, BodyPart.RIGHT_WRIST.value
        ]):
            right_shoulder = landmark_dict[BodyPart.RIGHT_SHOULDER.value]
            right_elbow = landmark_dict[BodyPart.RIGHT_ELBOW.value]
            right_wrist = landmark_dict[BodyPart.RIGHT_WRIST.value]
            
            angles["right_elbow"] = self._calculate_angle(
                (right_shoulder["x"], right_shoulder["y"]),
                (right_elbow["x"], right_elbow["y"]),
                (right_wrist["x"], right_wrist["y"])
            )
        
        # 计算膝盖角度
        if all(k in landmark_dict for k in [
            BodyPart.LEFT_HIP.value, BodyPart.LEFT_KNEE.value, BodyPart.LEFT_ANKLE.value
        ]):
            left_hip = landmark_dict[BodyPart.LEFT_HIP.value]
            left_knee = landmark_dict[BodyPart.LEFT_KNEE.value]
            left_ankle = landmark_dict[BodyPart.LEFT_ANKLE.value]
            
            angles["left_knee"] = self._calculate_angle(
                (left_hip["x"], left_hip["y"]),
                (left_knee["x"], left_knee["y"]),
                (left_ankle["x"], left_ankle["y"])
            )
        
        if all(k in landmark_dict for k in [
            BodyPart.RIGHT_HIP.value, BodyPart.RIGHT_KNEE.value, BodyPart.RIGHT_ANKLE.value
        ]):
            right_hip = landmark_dict[BodyPart.RIGHT_HIP.value]
            right_knee = landmark_dict[BodyPart.RIGHT_KNEE.value]
            right_ankle = landmark_dict[BodyPart.RIGHT_ANKLE.value]
            
            angles["right_knee"] = self._calculate_angle(
                (right_hip["x"], right_hip["y"]),
                (right_knee["x"], right_knee["y"]),
                (right_ankle["x"], right_ankle["y"])
            )
        
        return angles
    
    @staticmethod
    def _calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
        """
        计算三点之间的角度
        
        Args:
            a: 第一个点(x,y)
            b: 中间点(x,y)
            c: 第三个点(x,y)
            
        Returns:
            角度(度)
        """
        # 转换为numpy数组以便计算
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # 计算向量
        ba = a - b
        bc = c - b
        
        # 计算点积
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # 避免浮点误差
        
        # 计算角度
        angle = np.arccos(cosine_angle)
        
        # 转换为度
        angle = np.degrees(angle)
        
        return float(angle)

class HumanPoseDetector:
    """人体姿态检测类"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size: int = 100):
        """
        初始化人体姿态检测器
        
        Args:
            cache_dir: 缓存目录，None表示不缓存
            max_cache_size: 最大缓存条目数
        """
        self._pose_model = None
        self._face_feature_extractor = FaceFeatureExtractor()
        self._cache_dir = cache_dir
        
        # 创建缓存目录
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        # 配置模型参数
        self._model_complexity = {
            PoseDetectionLevel.FAST: 0,
            PoseDetectionLevel.STANDARD: 1,
            PoseDetectionLevel.ACCURATE: 2
        }
    
    def process_image(self, 
                      image_path: str,
                      detection_level: PoseDetectionLevel = PoseDetectionLevel.STANDARD,
                      face_feature_level: FaceFeatureLevel = FaceFeatureLevel.STANDARD,
                      min_detection_confidence: float = 0.5,
                      enable_cache: bool = True) -> Dict[str, Any]:
        """
        处理图像并检测人体姿态
        
        Args:
            image_path: 图像路径
            detection_level: 检测精度级别
            face_feature_level: 人脸特征提取精度级别
            min_detection_confidence: 最小检测置信度
            enable_cache: 是否启用缓存
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        # 生成缓存键
        cache_key = f"{os.path.basename(image_path)}_{detection_level}_{face_feature_level}_{min_detection_confidence}"
        
        # 尝试从缓存读取
        if enable_cache and self._cache_dir:
            cached_result = self._load_from_cache(cache_key)
            if cached_result:
                return cached_result
        
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            return {
                "success": False,
                "error": f"无法读取图像: {image_path}",
                "process_time": 0
            }
        
        # 初始化结果字典
        result = {
            "success": True,
            "image_path": image_path,
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0],
                "channels": img.shape[2]
            },
            "process_time": 0
        }
        
        # 检测图像类型
        image_type = self._detect_image_type(img)
        result["image_type"] = image_type
        
        # 根据图像类型和检测级别进行处理
        if image_type in [ImageType.PORTRAIT, ImageType.FULL_BODY, ImageType.GROUP]:
            # 处理人脸
            face_result = self._process_face(img, face_feature_level)
            if face_result:
                result["face_detection"] = face_result
            
            # 处理人体姿态
            pose_result = self._process_pose(img, detection_level, min_detection_confidence)
            if pose_result:
                result["pose_detection"] = pose_result
                
                # 如果检测到姿态，生成可视化结果
                if pose_result.get("pose_detected", False):
                    # 如果是精确模式，则处理人体分割
                    if detection_level == PoseDetectionLevel.ACCURATE:
                        segmentation = self._process_body_segmentation(img)
                        if segmentation:
                            result["body_segmentation"] = segmentation
        
        # 记录处理时间
        process_time = time.time() - start_time
        result["process_time"] = process_time
        
        # 保存到缓存
        if enable_cache and self._cache_dir:
            self._save_to_cache(cache_key, result)
        
        return result
    
    def process_video(self, 
                     video_path: str,
                     output_path: Optional[str] = None,
                     detection_level: PoseDetectionLevel = PoseDetectionLevel.FAST,
                     sample_rate: int = 5,
                     max_frames: int = 300) -> Dict[str, Any]:
        """
        处理视频并检测人体姿态
        
        Args:
            video_path: 视频路径
            output_path: 输出视频路径，None表示不生成输出视频
            detection_level: 检测精度级别
            sample_rate: 采样率(每N帧处理一次)
            max_frames: 最大处理帧数
            
        Returns:
            处理结果字典
        """
        start_time = time.time()
        
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {
                "success": False,
                "error": f"无法打开视频: {video_path}",
                "process_time": 0
            }
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化结果
        result = {
            "success": True,
            "video_path": video_path,
            "video_info": {
                "fps": fps,
                "width": frame_width,
                "height": frame_height,
                "total_frames": total_frames,
                "duration": total_frames / fps if fps > 0 else 0
            },
            "frame_results": [],
            "process_time": 0
        }
        
        # 设置输出视频
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_path, fourcc, fps/sample_rate, (frame_width, frame_height))
            result["output_video_path"] = output_path
        
        # 使用MediaPipe Pose模型
        model_complexity = self._model_complexity.get(detection_level, 1)
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as pose_model:
            frame_idx = 0
            processed_count = 0
            
            while cap.isOpened() and processed_count < max_frames:
                success, frame = cap.read()
                
                if not success:
                    break
                
                # 根据采样率处理帧
                if frame_idx % sample_rate == 0:
                    # 处理当前帧
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose_model.process(frame_rgb)
                    
                    # 检测是否有姿态
                    has_pose = pose_results.pose_landmarks is not None
                    
                    # 创建可视化
                    annotated_frame = frame.copy()
                    if has_pose:
                        mp_drawing.draw_landmarks(
                            annotated_frame,
                            pose_results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS
                        )
                    
                    # 保存到输出视频
                    if out:
                        out.write(annotated_frame)
                    
                    # 提取关键点
                    landmarks = []
                    if has_pose:
                        for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                            landmarks.append({
                                "idx": idx,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                                "visibility": landmark.visibility
                            })
                    
                    # 添加到结果
                    frame_result = {
                        "frame_idx": frame_idx,
                        "timestamp": frame_idx / fps,
                        "has_pose": has_pose,
                        "landmarks": landmarks if has_pose else []
                    }
                    
                    result["frame_results"].append(frame_result)
                    processed_count += 1
                
                frame_idx += 1
                
                # 检查是否达到最大帧数
                if processed_count >= max_frames:
                    break
        
        # 清理资源
        cap.release()
        if out:
            out.release()
        
        # 记录处理时间
        result["process_time"] = time.time() - start_time
        result["processed_frames"] = processed_count
        
        return result
    
    def _detect_image_type(self, img: np.ndarray) -> ImageType:
        """
        检测图像类型
        
        Args:
            img: 输入图像
            
        Returns:
            图像类型
        """
        # 调整图像大小以加快处理速度
        max_dim = 300
        h, w = img.shape[:2]
        
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            resized = cv2.resize(img, (int(w * scale), int(h * scale)))
        else:
            resized = img.copy()
            
        # 转换为RGB (MediaPipe需要RGB)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 使用MediaPipe Holistic模型检测人体和人脸
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=0,  # 使用低复杂度以加快速度
            min_detection_confidence=0.3
        ) as holistic:
            results = holistic.process(rgb_img)
            
            # 检查是否检测到人脸
            face_detected = results.face_landmarks is not None
            
            # 检查是否检测到姿态关键点
            pose_detected = results.pose_landmarks is not None
            
            # 计算可见的姿态关键点数量
            visible_pose_landmarks = 0
            if pose_detected:
                for landmark in results.pose_landmarks.landmark:
                    if landmark.visibility > 0.5:
                        visible_pose_landmarks += 1
            
            # 根据检测结果确定图像类型
            if face_detected and pose_detected:
                if visible_pose_landmarks > 20:  # 如果大部分姿态关键点可见
                    return ImageType.FULL_BODY
                elif visible_pose_landmarks > 5:  # 如果部分姿态关键点可见
                    return ImageType.PORTRAIT
            elif face_detected:
                # 使用简单的人脸计数方法确定是人像还是人群
                face_count = self._count_faces(resized)
                if face_count > 1:
                    return ImageType.GROUP
                else:
                    return ImageType.PORTRAIT
            
            # 如果没有检测到人脸和姿态，判断是文本、产品还是风景
            # 使用简单的启发式方法
            
            # 转为灰度图
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # 检测边缘
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 颜色方差(产品图像通常有较高的色彩对比度)
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            color_variance = np.var(hsv[:,:,1])  # 饱和度通道的方差
            
            # 文本检测(文本通常有高边缘密度和低色彩变化)
            if edge_density > 0.1 and color_variance < 300:
                return ImageType.TEXT
            
            # 产品检测(产品通常有中等边缘密度和中等色彩变化)
            elif 0.05 < edge_density < 0.15 and 300 < color_variance < 1000:
                return ImageType.PRODUCT
            
            # 风景检测(风景通常有低边缘密度和高色彩变化)
            elif edge_density < 0.1 and color_variance > 500:
                return ImageType.LANDSCAPE
                
            # 默认返回未知类型
            return ImageType.UNKNOWN
    
    def _count_faces(self, img: np.ndarray) -> int:
        """
        计算图像中的人脸数量
        
        Args:
            img: 输入图像
            
        Returns:
            人脸数量
        """
        # 转为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用Haar级联分类器检测人脸
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        return len(faces)
    
    def _process_face(self, img: np.ndarray, level: FaceFeatureLevel) -> Dict[str, Any]:
        """
        处理人脸
        
        Args:
            img: 输入图像
            level: 人脸特征提取精度级别
            
        Returns:
            人脸处理结果
        """
        # 使用FaceFeatureExtractor处理人脸
        face_features = self._face_feature_extractor.extract_face_features(img, level)
        return face_features
    
    def _process_pose(self, img: np.ndarray, 
                     level: PoseDetectionLevel, 
                     min_detection_confidence: float) -> Dict[str, Any]:
        """
        处理人体姿态
        
        Args:
            img: 输入图像
            level: 姿态检测精度级别
            min_detection_confidence: 最小检测置信度
            
        Returns:
            姿态处理结果
        """
        # 选择模型复杂度
        model_complexity = self._model_complexity.get(level, 1)
        
        # 转换为RGB (MediaPipe需要RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 初始化结果
        result = {
            "pose_detected": False,
            "detection_level": level
        }
        
        # 使用MediaPipe Pose
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=model_complexity,
            enable_segmentation=level == PoseDetectionLevel.ACCURATE,
            min_detection_confidence=min_detection_confidence
        ) as pose:
            # 处理图像
            pose_results = pose.process(rgb_img)
            
            if not pose_results.pose_landmarks:
                return result
            
            # 提取姿态关键点
            landmarks = []
            
            for idx, landmark in enumerate(pose_results.pose_landmarks.landmark):
                # 转换为像素坐标
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                
                # 添加到列表
                landmarks.append({
                    "idx": idx,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                    "px": px,
                    "py": py
                })
            
            # 计算身体边界框
            x_coords = [lm["px"] for lm in landmarks if lm["visibility"] > 0.5]
            y_coords = [lm["py"] for lm in landmarks if lm["visibility"] > 0.5]
            
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 添加边距
                margin_x = int((x_max - x_min) * 0.1)
                margin_y = int((y_max - y_min) * 0.1)
                
                x_min = max(0, x_min - margin_x)
                y_min = max(0, y_min - margin_y)
                x_max = min(w, x_max + margin_x)
                y_max = min(h, y_max + margin_y)
                
                width = x_max - x_min
                height = y_max - y_min
                
                # 相对坐标
                x_rel = x_min / w
                y_rel = y_min / h
                width_rel = width / w
                height_rel = height / h
                
                # 更新结果
                result.update({
                    "pose_detected": True,
                    "landmarks": landmarks,
                    "bbox": {
                        "x": x_min,
                        "y": y_min,
                        "width": width,
                        "height": height,
                        "x_rel": x_rel,
                        "y_rel": y_rel,
                        "width_rel": width_rel,
                        "height_rel": height_rel
                    }
                })
                
                # 如果启用了分割，处理分割掩码
                if pose_results.segmentation_mask is not None:
                    mask = pose_results.segmentation_mask
                    
                    # 创建分割图像
                    mask_binary = (mask > 0.1).astype(np.uint8) * 255
                    mask_path = str(Path(str(uuid.uuid4())).with_suffix('.pose_mask.png'))
                    cv2.imwrite(mask_path, mask_binary)
                    
                    result["segmentation_mask_path"] = mask_path
                
                # 生成姿态可视化
                annotated_img = img.copy()
                mp_drawing.draw_landmarks(
                    annotated_img,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                
                # 保存可视化
                vis_path = str(Path(str(uuid.uuid4())).with_suffix('.pose_vis.jpg'))
                cv2.imwrite(vis_path, annotated_img)
                result["visualization_path"] = vis_path
                
                # 进行姿态分析
                result["pose_analysis"] = self._analyze_pose(landmarks)
            
            return result
    
    def _process_body_segmentation(self, img: np.ndarray) -> Dict[str, Any]:
        """
        处理人体分割
        
        Args:
            img: 输入图像
            
        Returns:
            分割结果
        """
        # 转换为RGB (MediaPipe需要RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 初始化结果
        result = {}
        
        # 使用MediaPipe Selfie Segmentation
        selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # 处理图像
        segmentation_result = selfie_segmentation.process(rgb_img)
        
        # 如果成功获取分割掩码
        if segmentation_result.segmentation_mask is not None:
            mask = segmentation_result.segmentation_mask
            
            # 创建前景和背景图像
            condition = np.stack((mask,) * 3, axis=-1) > 0.1
            background = np.zeros(img.shape, dtype=np.uint8)
            background[:] = (192, 192, 192)  # 灰色背景
            
            foreground = np.where(condition, img, background)
            
            # 保存掩码和结果图像
            mask_binary = (mask > 0.1).astype(np.uint8) * 255
            mask_path = str(Path(str(uuid.uuid4())).with_suffix('.body_mask.png'))
            cv2.imwrite(mask_path, mask_binary)
            
            # 保存分割图像
            segmented_path = str(Path(str(uuid.uuid4())).with_suffix('.body_segmented.jpg'))
            cv2.imwrite(segmented_path, foreground)
            
            result.update({
                "segmentation_mask_path": mask_path,
                "segmented_image_path": segmented_path
            })
        
        return result
    
    def _analyze_pose(self, landmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析姿态信息
        
        Args:
            landmarks: 姿态关键点列表
            
        Returns:
            姿态分析结果
        """
        # 初始化结果
        result = {}
        
        # 提取关键关节点
        # MediaPipe Pose关键点映射:
        # 0: nose, 11-12: shoulders, 23-24: hips, 13-14: elbows, 15-16: wrists, 25-26: knees, 27-28: ankles
        
        # 查找特定关键点
        nose = next((lm for lm in landmarks if lm["idx"] == 0), None)
        left_shoulder = next((lm for lm in landmarks if lm["idx"] == 11), None)
        right_shoulder = next((lm for lm in landmarks if lm["idx"] == 12), None)
        left_hip = next((lm for lm in landmarks if lm["idx"] == 23), None)
        right_hip = next((lm for lm in landmarks if lm["idx"] == 24), None)
        left_knee = next((lm for lm in landmarks if lm["idx"] == 25), None)
        right_knee = next((lm for lm in landmarks if lm["idx"] == 26), None)
        left_ankle = next((lm for lm in landmarks if lm["idx"] == 27), None)
        right_ankle = next((lm for lm in landmarks if lm["idx"] == 28), None)
        
        # 计算姿势类型
        pose_type = self._determine_pose_type(nose, left_shoulder, right_shoulder, 
                                              left_hip, right_hip, left_knee, 
                                              right_knee, left_ankle, right_ankle)
        if pose_type:
            result["pose_type"] = pose_type
        
        # 计算身体比例
        body_proportions = self._calculate_body_proportions(nose, left_shoulder, right_shoulder, 
                                                          left_hip, right_hip, left_ankle, right_ankle)
        if body_proportions:
            result["body_proportions"] = body_proportions
        
        # 计算关节角度
        joint_angles = self._calculate_joint_angles(landmarks)
        if joint_angles:
            result["joint_angles"] = joint_angles
        
        return result
    
    def _determine_pose_type(self, nose, left_shoulder, right_shoulder, 
                           left_hip, right_hip, left_knee, 
                           right_knee, left_ankle, right_ankle) -> Optional[str]:
        """
        确定姿势类型
        
        Args:
            姿态关键点
            
        Returns:
            姿势类型
        """
        # 检查关键点是否有效
        keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
        if not all(keypoints):
            return None
        
        # 计算关键指标
        # 肩膀中点
        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        # 臀部中点
        hip_mid_y = (left_hip["y"] + right_hip["y"]) / 2
        
        # 躯干垂直度 (肩膀与臀部的垂直距离)
        torso_height = hip_mid_y - shoulder_mid_y
        
        # 腿部可见性
        legs_visible = left_knee and right_knee and left_ankle and right_ankle
        
        # 确定姿势类型
        if torso_height < 0.05:  # 躯干几乎水平
            return "lying"
        elif not legs_visible:  # 腿部不可见
            if nose and nose["y"] < shoulder_mid_y:  # 头部在躯干上方
                return "upper_body"
            else:
                return "unknown"
        else:  # 腿部可见
            # 检查站立或坐姿
            # 计算大腿角度
            if left_knee and left_hip:
                left_thigh_vertical = abs(left_knee["y"] - left_hip["y"])
            else:
                left_thigh_vertical = 0
                
            if right_knee and right_hip:
                right_thigh_vertical = abs(right_knee["y"] - right_hip["y"])
            else:
                right_thigh_vertical = 0
                
            thigh_vertical = (left_thigh_vertical + right_thigh_vertical) / 2
            
            if thigh_vertical < 0.1:  # 大腿接近水平
                return "sitting"
            else:
                return "standing"
    
    def _calculate_body_proportions(self, nose, left_shoulder, right_shoulder, 
                                  left_hip, right_hip, left_ankle, right_ankle) -> Dict[str, float]:
        """
        计算身体比例
        
        Args:
            姿态关键点
            
        Returns:
            身体比例字典
        """
        proportions = {}
        
        # 检查关键点是否有效
        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return proportions
        
        # 计算关键距离
        # 肩宽
        shoulder_width = abs(left_shoulder["x"] - right_shoulder["x"])
        # 臀宽
        hip_width = abs(left_hip["x"] - right_hip["x"])
        
        # 躯干高度
        torso_height = abs((left_hip["y"] + right_hip["y"]) / 2 - (left_shoulder["y"] + right_shoulder["y"]) / 2)
        
        # 腿长 (如果可见)
        leg_length = 0
        if left_hip and left_ankle:
            leg_length = abs(left_ankle["y"] - left_hip["y"])
        elif right_hip and right_ankle:
            leg_length = abs(right_ankle["y"] - right_hip["y"])
        
        # 头部高度 (如果可见)
        head_height = 0
        if nose and left_shoulder and right_shoulder:
            head_height = abs(nose["y"] - (left_shoulder["y"] + right_shoulder["y"]) / 2)
        
        # 添加到结果
        if shoulder_width > 0:
            proportions["shoulder_width"] = shoulder_width
        
        if hip_width > 0:
            proportions["hip_width"] = hip_width
            
        if torso_height > 0:
            proportions["torso_height"] = torso_height
            
        if leg_length > 0:
            proportions["leg_length"] = leg_length
            
        if head_height > 0:
            proportions["head_height"] = head_height
        
        # 计算比例
        if torso_height > 0 and leg_length > 0:
            proportions["leg_torso_ratio"] = leg_length / torso_height
            
        if shoulder_width > 0 and hip_width > 0:
            proportions["shoulder_hip_ratio"] = shoulder_width / hip_width
        
        return proportions
    
    def _calculate_joint_angles(self, landmarks: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        计算关节角度
        
        Args:
            landmarks: 姿态关键点列表
            
        Returns:
            关节角度字典
        """
        angles = {}
        
        # MediaPipe姿态索引映射
        # 肩-肘-手腕
        shoulder_elbow_wrist = [(11, 13, 15), (12, 14, 16)]  # 左右
        # 髋-膝-踝
        hip_knee_ankle = [(23, 25, 27), (24, 26, 28)]  # 左右
        # 肩-髋-膝
        shoulder_hip_knee = [(11, 23, 25), (12, 24, 26)]  # 左右
        
        # 函数: 计算三点角度
        def calculate_angle(a, b, c):
            if not all([a, b, c]):
                return None
                
            # 创建向量
            ba = [a["x"] - b["x"], a["y"] - b["y"]]
            bc = [c["x"] - b["x"], c["y"] - b["y"]]
            
            # 计算点积
            dot_product = ba[0] * bc[0] + ba[1] * bc[1]
            
            # 计算向量长度
            ba_length = (ba[0]**2 + ba[1]**2)**0.5
            bc_length = (bc[0]**2 + bc[1]**2)**0.5
            
            # 避免除以零
            if ba_length * bc_length == 0:
                return None
                
            # 计算夹角的余弦值
            cos_angle = dot_product / (ba_length * bc_length)
            
            # 防止因浮点误差导致的域错误
            cos_angle = max(-1, min(1, cos_angle))
            
            # 计算角度(度)
            angle = np.degrees(np.arccos(cos_angle))
            
            return angle
        
        # 查找关键点索引的函数
        def find_landmark(idx):
            return next((lm for lm in landmarks if lm["idx"] == idx), None)
        
        # 计算肘部角度
        for i, (shoulder, elbow, wrist) in enumerate(shoulder_elbow_wrist):
            shoulder_point = find_landmark(shoulder)
            elbow_point = find_landmark(elbow)
            wrist_point = find_landmark(wrist)
            
            if all([shoulder_point, elbow_point, wrist_point]) and all([p["visibility"] > 0.5 for p in [shoulder_point, elbow_point, wrist_point]]):
                side = "left" if i == 0 else "right"
                angle = calculate_angle(shoulder_point, elbow_point, wrist_point)
                if angle is not None:
                    angles[f"{side}_elbow"] = round(angle, 1)
        
        # 计算膝盖角度
        for i, (hip, knee, ankle) in enumerate(hip_knee_ankle):
            hip_point = find_landmark(hip)
            knee_point = find_landmark(knee)
            ankle_point = find_landmark(ankle)
            
            if all([hip_point, knee_point, ankle_point]) and all([p["visibility"] > 0.5 for p in [hip_point, knee_point, ankle_point]]):
                side = "left" if i == 0 else "right"
                angle = calculate_angle(hip_point, knee_point, ankle_point)
                if angle is not None:
                    angles[f"{side}_knee"] = round(angle, 1)
        
        # 计算髋部角度
        for i, (shoulder, hip, knee) in enumerate(shoulder_hip_knee):
            shoulder_point = find_landmark(shoulder)
            hip_point = find_landmark(hip)
            knee_point = find_landmark(knee)
            
            if all([shoulder_point, hip_point, knee_point]) and all([p["visibility"] > 0.5 for p in [shoulder_point, hip_point, knee_point]]):
                side = "left" if i == 0 else "right"
                angle = calculate_angle(shoulder_point, hip_point, knee_point)
                if angle is not None:
                    angles[f"{side}_hip"] = round(angle, 1)
        
        return angles
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]) -> None:
        """
        保存结果到缓存
        
        Args:
            cache_key: 缓存键
            result: 结果字典
        """
        if not self._cache_dir:
            return
            
        # 创建缓存文件路径
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")
        
        try:
            # 移除无法JSON序列化的字段
            result_copy = result.copy()
            
            # 保存到文件
            with open(cache_file, 'w') as f:
                json.dump(result_copy, f)
        except Exception as e:
            print(f"缓存保存失败: {str(e)}")
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        从缓存中加载结果
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的结果字典，如果不存在则返回None
        """
        if not self._cache_dir:
            return None
            
        # 创建缓存文件路径
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")
        
        # 检查文件是否存在
        if not os.path.exists(cache_file):
            return None
            
        try:
            # 从文件加载
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"缓存加载失败: {str(e)}")
            return None 