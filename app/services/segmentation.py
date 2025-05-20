import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import time
import os
from enum import Enum
from PIL import Image
from app.core.config import settings
from app.utils.image_utils import ImageUtils

class SegmentationModel(str, Enum):
    """分割模型类型枚举"""
    U2NET = "u2net"      # U^2-Net模型
    MODNET = "modnet"    # MODNet模型

# U^2-Net 模型定义
class ConvBNRelu(nn.Module):
    def __init__(self, in_ch, out_ch, dirate=1):
        super(ConvBNRelu, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class RSU7(nn.Module):
    """U^2-Net中的RSU-7模块"""
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7, self).__init__()
        self.conv_b0 = ConvBNRelu(in_ch, out_ch)
        self.conv_b1 = ConvBNRelu(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv1 = ConvBNRelu(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = ConvBNRelu(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = ConvBNRelu(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4 = ConvBNRelu(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv5 = ConvBNRelu(mid_ch, mid_ch)
        self.pool6 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv6 = ConvBNRelu(mid_ch, mid_ch)
        self.conv7 = ConvBNRelu(mid_ch, mid_ch, dirate=2)
        self.conv6d = ConvBNRelu(mid_ch*2, mid_ch, dirate=1)
        self.conv5d = ConvBNRelu(mid_ch*2, mid_ch, dirate=1)
        self.conv4d = ConvBNRelu(mid_ch*2, mid_ch, dirate=1)
        self.conv3d = ConvBNRelu(mid_ch*2, mid_ch, dirate=1)
        self.conv2d = ConvBNRelu(mid_ch*2, mid_ch, dirate=1)
        self.conv1d = ConvBNRelu(mid_ch*2, out_ch, dirate=1)

    def forward(self, x):
        hx = x
        hxin = self.conv_b0(hx)
        hx1 = self.conv_b1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.conv1(hx)
        hx = self.pool2(hx2)
        hx3 = self.conv2(hx)
        hx = self.pool3(hx3)
        hx4 = self.conv3(hx)
        hx = self.pool4(hx4)
        hx5 = self.conv4(hx)
        hx = self.pool5(hx5)
        hx6 = self.conv5(hx)
        hx = self.pool6(hx6)
        hx7 = self.conv6(hx)
        hx7 = self.conv7(hx7)
        
        hx6d = self.conv6d(torch.cat((hx7, hx6), 1))
        hx5d = self.conv5d(torch.cat((hx6d, hx5), 1))
        hx4d = self.conv4d(torch.cat((hx5d, hx4), 1))
        hx3d = self.conv3d(torch.cat((hx4d, hx3), 1))
        hx2d = self.conv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.conv1d(torch.cat((hx2d, hx1), 1))
        
        return hx1d + hxin

# 简化版本的U^2-Net模型
class U2NET(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        
        # 简化：仅使用一个RSU模块
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage2 = RSU7(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        self.stage3 = RSU7(128, 64, 256)
        
        # 简化：只有一个上采样阶段
        self.stage2d = RSU7(256, 64, 128)
        self.stage1d = RSU7(128, 32, 64)
        
        # 输出卷积
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        
    def forward(self, x):
        # 编码器
        hx1 = self.stage1(x)
        hx = self.pool12(hx1)
        
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        
        hx3 = self.stage3(hx)
        
        # 解码器
        hx2d = self.stage2d(hx3)
        hx2d = F.interpolate(hx2d, scale_factor=2, mode='bilinear', align_corners=True)
        
        hx1d = self.stage1d(hx2d)
        
        # 侧输出
        d1 = self.side1(hx1d)
        
        # 归一化
        d1 = torch.sigmoid(d1)
        
        return d1

# MODNet模型 (简化版本)
class MODNet(nn.Module):
    def __init__(self):
        super(MODNet, self).__init__()
        # 这里使用简化实现，实际项目中应该加载完整的MODNet架构
        
        # 编码器 (类似MobileNetV2)
        self.encoder = nn.Sequential(
            ConvBNRelu(3, 32, dirate=1),
            ConvBNRelu(32, 64, dirate=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            ConvBNRelu(64, 128, dirate=1),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            ConvBNRelu(128, 64, dirate=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNRelu(64, 32, dirate=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        feat = self.encoder(x)
        pred_matte = self.decoder(feat)
        return pred_matte

class BackgroundSegmentationService:
    """背景分割服务，用于图像前景/背景分割"""
    
    def __init__(self):
        """初始化背景分割服务"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_utils = ImageUtils()
        self.u2net_model = None
        self.modnet_model = None
        
        # 定义预处理转换
        self.u2net_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        self.modnet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                std=[0.5, 0.5, 0.5])
        ])
    
    def load_model(self, model_type: SegmentationModel = SegmentationModel.U2NET):
        """加载分割模型"""
        if model_type == SegmentationModel.U2NET and self.u2net_model is None:
            print(f"Loading U2NET model from {settings.MODEL_PATHS['u2net']}")
            
            # 创建模型实例
            self.u2net_model = U2NET()
            
            # 加载预训练权重
            if os.path.exists(settings.MODEL_PATHS["u2net"]):
                try:
                    self.u2net_model.load_state_dict(torch.load(settings.MODEL_PATHS["u2net"], map_location=self.device))
                    self.u2net_model = self.u2net_model.to(self.device)
                    self.u2net_model.eval()
                    print("U2NET model loaded successfully")
                except Exception as e:
                    print(f"Failed to load U2NET model: {e}")
                    # 如果加载失败，仍然返回未初始化的模型以便演示
            else:
                print(f"U2NET model not found at {settings.MODEL_PATHS['u2net']}, using uninitialized model")
            
        elif model_type == SegmentationModel.MODNET and self.modnet_model is None:
            print(f"Loading MODNet model from {settings.MODEL_PATHS['modnet']}")
            
            # 创建模型实例
            self.modnet_model = MODNet()
            
            # 加载预训练权重
            if os.path.exists(settings.MODEL_PATHS["modnet"]):
                try:
                    self.modnet_model.load_state_dict(torch.load(settings.MODEL_PATHS["modnet"], map_location=self.device))
                    self.modnet_model = self.modnet_model.to(self.device)
                    self.modnet_model.eval()
                    print("MODNet model loaded successfully")
                except Exception as e:
                    print(f"Failed to load MODNet model: {e}")
                    # 如果加载失败，仍然返回未初始化的模型以便演示
            else:
                print(f"MODNet model not found at {settings.MODEL_PATHS['modnet']}, using uninitialized model")
    
    def segment_foreground(self, image_path: str, model_type: SegmentationModel = SegmentationModel.U2NET) -> Dict[str, Any]:
        """
        执行前景分割，生成alpha掩码
        
        Args:
            image_path: 输入图像路径
            model_type: 使用的分割模型类型
            
        Returns:
            包含分割结果的字典，包括alpha掩码和处理时间
        """
        start_time = time.time()
        
        # 加载模型
        self.load_model(model_type)
        
        # 读取并预处理图像
        img = self.image_utils.read_image(image_path)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整图像大小，保持纵横比，提高处理速度
        h, w = img.shape[:2]
        max_size = 512  # 为了处理速度，限制最大尺寸
        
        if max(h, w) > max_size:
            if h > w:
                new_h, new_w = max_size, int(w * max_size / h)
            else:
                new_h, new_w = int(h * max_size / w), max_size
            
            # 确保尺寸是32的倍数（有些模型需要）
            new_h = (new_h // 32) * 32
            new_w = (new_w // 32) * 32
            
            resized_img = cv2.resize(rgb_img, (new_w, new_h))
        else:
            resized_img = rgb_img
        
        # 图像转换为PIL Image(某些转换需要)
        pil_img = Image.fromarray(resized_img)
        
        # 根据选择的模型进行预处理
        if model_type == SegmentationModel.U2NET:
            # U^2-Net预处理和推理
            input_tensor = self.u2net_transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.u2net_model(input_tensor)
                pred = output.squeeze().cpu().numpy()
        else:
            # MODNet预处理和推理
            input_tensor = self.modnet_transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.modnet_model(input_tensor)
                pred = output.squeeze().cpu().numpy()
        
        # 处理预测结果
        mask = (pred * 255).astype(np.uint8)
        
        # 将掩码调整回原始图像大小
        mask = cv2.resize(mask, (w, h))
        
        # 后处理掩码，确保平滑过渡
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # 计算处理时间
        process_time = time.time() - start_time
        
        # 保存alpha掩码
        mask_path = str(Path(image_path).with_suffix('.alpha_mask.png'))
        cv2.imwrite(mask_path, mask)
        
        return {
            "mask_path": mask_path,
            "original_image_path": image_path,
            "process_time": process_time
        }
    
    def remove_background(self, image_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        移除背景并可选地替换为纯色或其他图像
        
        Args:
            image_path: 输入图像路径
            options: 配置选项，包括：
                     - model_type: 使用的模型类型
                     - replace_background: 是否替换背景
                     - background_color: 背景颜色
                     - background_image_url: 背景图像URL
                     - foreground_boost: 前景增强因子(0-1)
                     - edge_refinement: 边缘细化程度(0-1)
                     
        Returns:
            包含处理结果的字典
        """
        # 解析选项
        model_type_str = options.get("model_type", SegmentationModel.U2NET.value)
        model_type = SegmentationModel(model_type_str)
        replace_background = options.get("replace_background", False)
        background_color = options.get("background_color")
        bg_image_path = options.get("background_image_url")
        foreground_boost = min(max(options.get("foreground_boost", 0.0), 0.0), 1.0)
        edge_refinement = min(max(options.get("edge_refinement", 0.5), 0.0), 1.0)
        
        # 执行前景分割
        segmentation_result = self.segment_foreground(image_path, model_type)
        mask_path = segmentation_result["mask_path"]
        
        # 读取原始图像和掩码
        img = self.image_utils.read_image(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 前景增强
        if foreground_boost > 0:
            # 基于前景增强因子调整掩码
            boost_factor = 1.0 + foreground_boost
            mask = np.clip(mask * boost_factor, 0, 255).astype(np.uint8)
        
        # 边缘细化
        if edge_refinement > 0:
            # 使用不同大小的高斯核进行边缘平滑
            blur_size = int(5 + edge_refinement * 10)
            if blur_size % 2 == 0:
                blur_size += 1  # 确保是奇数
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # 创建包含alpha通道的图像
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = mask
        
        # 如果需要替换背景
        if replace_background:
            if bg_image_path:
                # 读取背景图像
                bg_img = self.image_utils.read_image(bg_image_path)
                # 确保背景图像和原图尺寸一致
                bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))
                
                # 创建前景和背景掩码
                fg_mask = mask / 255.0
                fg_mask = np.stack([fg_mask] * 3, axis=2)
                bg_mask = 1.0 - fg_mask
                
                # 混合图像
                output_img = img * fg_mask + bg_img * bg_mask
                output_img = output_img.astype(np.uint8)
                
                # 保存结果
                result_path = str(Path(image_path).with_suffix('.segmented_bg.jpg'))
                self.image_utils.save_image(output_img, result_path)
                
                return {
                    "result_path": result_path,
                    "mask_path": mask_path,
                    "has_alpha": False,
                    "process_time": segmentation_result["process_time"]
                }
                
            elif background_color:
                if background_color == "transparent":
                    # 保持透明背景
                    result_path = str(Path(image_path).with_suffix('.transparent.png'))
                    cv2.imwrite(result_path, rgba)
                    
                    return {
                        "result_path": result_path,
                        "mask_path": mask_path,
                        "has_alpha": True,
                        "process_time": segmentation_result["process_time"]
                    }
                else:
                    # 解析颜色 (如 "#ff0000")
                    color = background_color.lstrip('#')
                    b, g, r = tuple(int(color[i:i+2], 16) for i in (4, 2, 0))
                    
                    # 创建纯色背景
                    bg = np.ones_like(img) * np.array([b, g, r], dtype=np.uint8)
                    
                    # 创建前景掩码
                    fg_mask = mask / 255.0
                    fg_mask = np.stack([fg_mask] * 3, axis=2)
                    
                    # 创建背景掩码
                    bg_mask = 1.0 - fg_mask
                    
                    # 混合图像
                    output_img = img * fg_mask + bg * bg_mask
                    output_img = output_img.astype(np.uint8)
                    
                    # 保存结果
                    result_path = str(Path(image_path).with_suffix('.segmented_color.jpg'))
                    self.image_utils.save_image(output_img, result_path)
                    
                    return {
                        "result_path": result_path,
                        "mask_path": mask_path,
                        "has_alpha": False,
                        "process_time": segmentation_result["process_time"]
                    }
        
        # 默认返回透明背景图像
        result_path = str(Path(image_path).with_suffix('.transparent.png'))
        cv2.imwrite(result_path, rgba)
        
        return {
            "result_path": result_path,
            "mask_path": mask_path,
            "has_alpha": True,
            "process_time": segmentation_result["process_time"]
        } 