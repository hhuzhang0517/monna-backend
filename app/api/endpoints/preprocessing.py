from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Path, Form
from typing import Optional, Dict, Any, List
import json
from enum import Enum
from app.models.schemas import ImageUploadResponse, TaskStatus
from app.utils.storage import StorageService
from app.services.preprocessing import ImagePreprocessingService, ProcessingMode
from app.worker.celery_app import celery_app
from app.worker.tasks import process_image_task
from app.core.config import settings

router = APIRouter()

class PreprocessingModeAPI(str, Enum):
    """API使用的预处理模式枚举"""
    BASIC = "basic"        # 仅进行基本处理
    FACE = "face"          # 人脸处理
    FULL = "full"          # 完整处理

@router.post("/process", response_model=ImageUploadResponse)
async def process_image(
    file: UploadFile = File(...),
    mode: PreprocessingModeAPI = Query(PreprocessingModeAPI.FULL, description="预处理模式")
):
    """
    图像预处理
    - file: 要处理的图像文件
    - mode: 预处理模式 (basic: 基本处理, face: 人脸处理, full: 完整处理)
    """
    # 验证文件类型
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400, 
            detail="仅支持JPG和PNG格式图像"
        )
        
    # 验证文件大小
    content = await file.read()
    await file.seek(0)  # 重置文件指针
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件大小超过限制 ({settings.MAX_FILE_SIZE // (1024*1024)}MB)"
        )
    
    # 保存上传的图像
    storage = StorageService()
    image_path = await storage.save_upload(file, "preprocessing")
    
    # 选择预处理模式
    processing_mode = ProcessingMode(mode.value)
    
    # 创建任务
    task = process_image_task.delay(image_path, processing_mode.value)
    
    # 返回任务ID
    return ImageUploadResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message="图像预处理任务已提交"
    )

@router.get("/result/{task_id}")
async def get_preprocessing_result(
    task_id: str = Path(..., description="任务ID")
):
    """
    获取预处理结果
    - task_id: 预处理任务ID
    """
    # 获取任务
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        return {"status": "pending", "message": "任务正在等待处理"}
    elif task.state == 'PROCESSING':
        return {"status": "processing", "message": "任务正在处理中"}
    elif task.state == 'SUCCESS':
        # 获取任务结果
        result = task.result
        
        if not result:
            return {"status": "failed", "message": "预处理失败，未返回结果"}
        
        # 过滤敏感信息和路径，增加可访问URL
        filtered_result = {}
        for key, value in result.items():
            # 跳过预处理路径等服务器路径信息
            if key.endswith('_path') and isinstance(value, str):
                # 替换为可访问的URL路径
                filtered_result[key.replace('_path', '_url')] = storage.get_file_url(value)
            else:
                filtered_result[key] = value
            
        return {
            "status": "success",
            "task_id": task_id,
            "result": filtered_result
        }
    else:
        return {"status": "failed", "message": "任务处理失败"} 