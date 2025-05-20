from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form, Query, Path
from typing import Optional, Dict, Any, List
import json
import re
import logging
from app.models.schemas import ImageUploadResponse, BackgroundRemovalOptions, BackgroundSegmentationOptions, CartoonOptions, TaskStatus, TaskStatusResponse
from app.utils.storage import StorageService
from app.worker.tasks import process_background_removal, process_background_segmentation, process_cartoonization, process_style_transfer
from app.worker.celery_app import celery_app
from app.core.config import settings
import shutil
import os
import time
import uuid
import asyncio
import cv2
import numpy as np
from pathlib import Path as PathLib

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(settings.BASE_DIR / "logs" / "api.log")),
        logging.StreamHandler()
    ]
)

# 确保日志目录存在
os.makedirs(settings.BASE_DIR / "logs", exist_ok=True)

router = APIRouter()

# 定义允许的颜色格式的正则表达式
COLOR_PATTERN = re.compile(r'^#(?:[0-9a-fA-F]{3}){1,2}$|^transparent$')

# 定义允许的风格类型
ALLOWED_CARTOON_STYLES = ["anime", "cartoon"]

# 定义允许的分割模型类型
ALLOWED_SEGMENTATION_MODELS = ["u2net", "modnet"]

# 定义允许的风格转换类型
ALLOWED_STYLE_TRANSFER_TYPES = ["wedding", "model", "oil-painting"]

# 创建必要的目录
os.makedirs(settings.UPLOAD_DIR / "style_transfer" / "wedding", exist_ok=True)
os.makedirs(settings.UPLOAD_DIR / "style_transfer" / "model", exist_ok=True)
os.makedirs(settings.RESULTS_DIR / "style_transfer", exist_ok=True)

# 简单的任务状态存储
tasks = {}

@router.post("/remove-background", response_model=ImageUploadResponse)
async def remove_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    replace_background: bool = Form(False),
    background_color: Optional[str] = Form(None),
    background_image: Optional[UploadFile] = File(None),
):
    """
    移除图像背景
    - replace_background: 是否替换背景
    - background_color: 背景颜色 (如 "#ff0000" 或 "transparent")
    - background_image: 背景图像文件 (可选)
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
    
    # 验证背景颜色格式
    if background_color and not COLOR_PATTERN.match(background_color):
        raise HTTPException(
            status_code=400,
            detail="背景颜色格式无效，请使用HEX格式 (如 #ff0000) 或 'transparent'"
        )
    
    # 保存上传的图像
    storage = StorageService()
    image_path = await storage.save_upload(file, "background")
    
    # 保存背景图像 (如果提供了)
    background_image_path = None
    if background_image:
        # 验证背景图像类型
        if background_image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail="背景图像仅支持JPG和PNG格式"
            )
        background_image_path = await storage.save_upload(background_image, "background/bg")
    
    # 准备选项
    options = {
        "replace_background": replace_background,
        "background_color": background_color
    }
    
    if background_image_path:
        options["background_image_url"] = background_image_path
    
    # 启动异步任务
    task = process_background_removal.delay(image_path, options)

    # 返回任务ID
    return ImageUploadResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message="背景去除任务已提交，正在处理中"
    )

@router.post("/segment-background", response_model=ImageUploadResponse)
async def segment_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    model_type: str = Form("u2net"),
    replace_background: bool = Form(False),
    background_color: Optional[str] = Form(None),
    background_image: Optional[UploadFile] = File(None),
    foreground_boost: float = Form(0.0),
    edge_refinement: float = Form(0.5)
):
    """
    使用AI模型分割图像前景/背景
    - model_type: 分割模型类型 ("u2net" 或 "modnet")
    - replace_background: 是否替换背景
    - background_color: 背景颜色 (如 "#ff0000" 或 "transparent")
    - background_image: 背景图像文件 (可选)
    - foreground_boost: 前景增强因子 (0-1)
    - edge_refinement: 边缘细化程度 (0-1)
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
    
    # 验证模型类型
    if model_type not in ALLOWED_SEGMENTATION_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的分割模型类型，可选值: {', '.join(ALLOWED_SEGMENTATION_MODELS)}"
        )
    
    # 验证背景颜色格式
    if background_color and not COLOR_PATTERN.match(background_color):
        raise HTTPException(
            status_code=400,
            detail="背景颜色格式无效，请使用HEX格式 (如 #ff0000) 或 'transparent'"
        )
    
    # 验证前景增强和边缘细化参数
    if not (0 <= foreground_boost <= 1):
        raise HTTPException(
            status_code=400,
            detail="前景增强因子必须在0-1范围内"
        )
        
    if not (0 <= edge_refinement <= 1):
        raise HTTPException(
            status_code=400,
            detail="边缘细化程度必须在0-1范围内"
        )
    
    # 保存上传的图像
    storage = StorageService()
    image_path = await storage.save_upload(file, "segmentation")
    
    # 保存背景图像 (如果提供了)
    background_image_path = None
    if background_image:
        # 验证背景图像类型
        if background_image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail="背景图像仅支持JPG和PNG格式"
            )
        background_image_path = await storage.save_upload(background_image, "segmentation/bg")
    
    # 准备选项
    options = {
        "model_type": model_type,
        "replace_background": replace_background,
        "background_color": background_color,
        "foreground_boost": foreground_boost,
        "edge_refinement": edge_refinement
    }
    
    if background_image_path:
        options["background_image_url"] = background_image_path
    
    # 启动异步任务
    task = process_background_segmentation.delay(image_path, options)

    # 返回任务ID
    return ImageUploadResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message="背景分割任务已提交，正在处理中"
    )

@router.post("/cartoonize", response_model=ImageUploadResponse)
async def cartoonize(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    style: str = Form("anime")  # "anime" 或 "cartoon"
):
    """
    照片卡通化
    - style: 卡通风格类型 ("anime" - 动漫风格, "cartoon" - 卡通风格)
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
    
    # 验证风格参数
    if style not in ALLOWED_CARTOON_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"风格类型无效，可选值: {', '.join(ALLOWED_CARTOON_STYLES)}"
        )
    
    # 保存上传的图像
    storage = StorageService()
    image_path = await storage.save_upload(file, "cartoon")
    
    # 准备选项
    options = {"style": style}
    
    # 启动异步任务
    task = process_cartoonization.delay(image_path, options)
    
    # 返回任务ID
    return ImageUploadResponse(
        task_id=task.id,
        status=TaskStatus.PENDING,
        message="卡通化任务已提交，正在处理中"
    )

@router.post("/style-transfer", summary="婚礼风格照片生成")
async def upload_for_style_transfer(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    style: str = Form("wedding")
):
    """
    上传图片进行风格转换
    
    - **files**: 图片文件列表 (4-8张照片)
    - **style**: 风格类型，目前支持 'wedding'
    """
    # 验证文件数量
    if len(files) < 4 or len(files) > 8:
        raise HTTPException(status_code=400, detail=f"请上传4-8张照片，当前上传了{len(files)}张")
    
    # 验证文件类型
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是图片格式")
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"接收到风格转换请求，创建任务 {task_id}")
    
    # 保存文件
    saved_files = []
    upload_dir = settings.UPLOAD_DIR / "style_transfer" / style
    
    for i, file in enumerate(files):
        # 生成唯一文件名
        filename = f"{task_id}_{i}{PathLib(file.filename).suffix}"
        file_path = upload_dir / filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        saved_files.append(str(file_path))
        logger.info(f"保存文件: {file_path}")
    
    # 初始化任务状态
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "created_at": time.time(),
        "files": saved_files,
        "results": [],
        "style": style
    }
    
    # 在后台处理任务
    background_tasks.add_task(process_style_transfer_mock, task_id, saved_files, style)
    
    return {
        "task_id": task_id,
        "message": "上传成功，开始处理",
        "status": "processing"
    }

@router.get("/tasks/{task_id}", summary="获取任务状态")
async def get_task_status(task_id: str):
    """
    获取任务处理状态
    
    - **task_id**: 任务ID
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "created_at": task["created_at"],
        "results": task["results"] if "results" in task else []
    }

# 模拟风格转换处理
async def process_style_transfer_mock(task_id: str, file_paths: List[str], style: str):
    """模拟风格转换处理过程"""
    logger.info(f"开始处理任务 {task_id}，文件数量: {len(file_paths)}")
    
    # 更新任务状态
    tasks[task_id]["status"] = "processing"
    
    # 模拟处理时间
    total_steps = len(file_paths) * 2
    step = 0
    
    for i, file_path in enumerate(file_paths):
        # 读取图像
        try:
            img = cv2.imread(file_path)
            if img is None:
                logger.error(f"无法读取图像: {file_path}")
                continue
            
            # 模拟处理第一步：调整大小
            step += 1
            tasks[task_id]["progress"] = int((step / total_steps) * 100)
            await asyncio.sleep(1)  # 模拟处理时间
            
            # 调整图像大小
            width = 800
            height = int(img.shape[0] * width / img.shape[1])
            img_resized = cv2.resize(img, (width, height))
            
            # 模拟处理第二步：应用滤镜效果
            step += 1
            tasks[task_id]["progress"] = int((step / total_steps) * 100)
            await asyncio.sleep(1)  # 模拟处理时间
            
            # 应用简单滤镜（例如，增强对比度和亮度）
            if style == "wedding":
                # 婚礼风格：温暖色调，轻微模糊
                img_processed = cv2.convertScaleAbs(img_resized, alpha=1.1, beta=30)
                img_processed = cv2.GaussianBlur(img_processed, (5, 5), 0)
                
                # 增加暖色调
                img_processed = img_processed.astype(np.float32)
                img_processed[:,:,0] *= 0.9  # 降低蓝色通道
                img_processed[:,:,2] *= 1.15  # 增强红色通道
                img_processed = np.clip(img_processed, 0, 255).astype(np.uint8)
            else:
                # 默认处理
                img_processed = cv2.convertScaleAbs(img_resized, alpha=1.2, beta=15)
            
            # 保存处理后的图像
            result_filename = f"{task_id}_result_{i}.jpg"
            result_path = settings.RESULTS_DIR / "style_transfer" / result_filename
            cv2.imwrite(str(result_path), img_processed)
            
            # 添加结果URL
            result_url = f"/api/v1/static/data/results/style_transfer/{result_filename}"
            if "results" not in tasks[task_id]:
                tasks[task_id]["results"] = []
            tasks[task_id]["results"].append(result_url)
            
            logger.info(f"处理完成图像 {i+1}/{len(file_paths)}, 结果: {result_path}")
            
        except Exception as e:
            logger.error(f"处理图像时出错: {e}", exc_info=True)
    
    # 完成处理
    tasks[task_id]["status"] = "completed"
    tasks[task_id]["progress"] = 100
    logger.info(f"任务 {task_id} 处理完成")

@router.delete("/tasks/{task_id}")
async def cancel_task(
    task_id: str = Path(..., description="任务ID")
):
    """
    取消任务
    - task_id: 要取消的任务ID
    """
    # 获取任务
    task = celery_app.AsyncResult(task_id)
    
    if task.state in ['PENDING', 'PROCESSING']:
        # 尝试终止任务
        task.revoke(terminate=True)
        return {"message": "任务已取消"}
    else:
        # 任务已经结束，无法取消
        raise HTTPException(
            status_code=400,
            detail="无法取消已完成或失败的任务"
        )

@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    limit: int = Query(10, ge=1, le=100, description="返回结果的最大数量"),
    status: Optional[TaskStatus] = Query(None, description="按状态筛选任务")
):
    """
    列出最近的任务
    - limit: 返回的最大任务数量
    - status: 可选，按状态筛选任务
    """
    # 在生产环境中，这应该从数据库获取任务列表
    # 这里提供一个简化的实现
    
    # 实际项目中，应该实现一个任务数据库或使用Celery的监控工具
    # 当前实现仅返回一个示例任务列表
    return [
        TaskStatusResponse(
            task_id="example-task-1",
            status=TaskStatus.COMPLETED,
            result_url="/static/results/example.jpg",
            message="示例任务",
            created_at=1617180000.0
        )
    ]