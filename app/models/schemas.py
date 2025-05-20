from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any
from enum import Enum
import time

class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ImageUploadResponse(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    message: str
    
class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result_url: Optional[str] = None  # 兼容单个结果的情况
    result_urls: Optional[List[str]] = None  # 支持多个结果URL
    mask_url: Optional[str] = None  # 分割掩码URL
    message: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    style: Optional[str] = None  # 风格类型
    count: Optional[int] = None  # 生成图片数量
    
class BackgroundRemovalOptions(BaseModel):
    replace_background: bool = False
    background_color: Optional[str] = None  # 如 "#ff0000" 或 "transparent"
    background_image_url: Optional[str] = None

class BackgroundSegmentationOptions(BaseModel):
    model_type: str = "u2net"  # "u2net" 或 "modnet"
    replace_background: bool = False
    background_color: Optional[str] = None  # 如 "#ff0000" 或 "transparent"
    background_image_url: Optional[str] = None
    foreground_boost: float = 0.0  # 前景增强因子(0-1)
    edge_refinement: float = 0.5  # 边缘细化程度(0-1)
    
class StyleTransferOptions(BaseModel):
    style_name: str  # 如 "古风油画", "漫画风" 等
    strength: float = Field(0.75, ge=0.0, le=1.0)  # 风格强度

class CartoonOptions(BaseModel):
    style: str = "anime"  # "anime" 或 "cartoon"
    
class VirtualTryonOptions(BaseModel):
    item_type: str  # "clothing", "earrings", "necklace" 等
    item_id: str    # 物品ID
    
class InpaintingOptions(BaseModel):
    mask: Optional[str] = None  # Base64编码的掩码图像
    prompt: Optional[str] = None  # 文本提示（用于Stable Diffusion）
    
class ExpressionOptions(BaseModel):
    target_expression: str  # "smile", "surprise", "angry" 等
    
class AgingOptions(BaseModel):
    target_age: int  # 目标年龄 (如 5, 20, 40, 70)

class ErrorResponse(BaseModel):
    detail: str