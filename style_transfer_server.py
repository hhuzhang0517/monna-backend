from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import shutil
import os
import logging
import time
import uuid
from pathlib import Path
import asyncio
import cv2
import numpy as np
import uvicorn

# 定义路径常量
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
RESULTS_DIR = BASE_DIR / "data" / "results"
LOGS_DIR = BASE_DIR / "logs"

# 创建必要的目录
os.makedirs(UPLOAD_DIR / "style_transfer" / "wedding", exist_ok=True)
os.makedirs(UPLOAD_DIR / "style_transfer" / "model", exist_ok=True)
os.makedirs(RESULTS_DIR / "style_transfer", exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(LOGS_DIR / "style_transfer.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("style_transfer_api")

# 创建FastAPI应用
app = FastAPI(title="婚礼风格照片生成API", description="简单的风格转换API")

# 简单的任务状态存储
tasks = {}

# 设置静态文件服务
app.mount("/static", StaticFiles(directory=str(BASE_DIR)), name="static")

@app.post("/api/v1/style-transfer", summary="婚礼风格照片生成")
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
    if len(files) < 1 or len(files) > 8:
        raise HTTPException(status_code=400, detail=f"请上传1-8张照片，当前上传了{len(files)}张")
    
    # 验证文件类型
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"文件 {file.filename} 不是图片格式")
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"接收到风格转换请求，创建任务 {task_id}")
    
    # 保存文件
    saved_files = []
    upload_dir = UPLOAD_DIR / "style_transfer" / style
    
    for i, file in enumerate(files):
        # 生成唯一文件名
        filename = f"{task_id}_{i}{Path(file.filename).suffix}"
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

@app.get("/api/v1/tasks/{task_id}", summary="获取任务状态")
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
            result_path = RESULTS_DIR / "style_transfer" / result_filename
            cv2.imwrite(str(result_path), img_processed)
            
            # 添加结果URL
            result_url = f"/static/data/results/style_transfer/{result_filename}"
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

@app.get("/")
async def root():
    """API根路径，返回简单的欢迎信息"""
    return {"message": "欢迎使用婚礼风格照片生成API", "docs_url": "/docs"}

if __name__ == "__main__":
    uvicorn.run("style_transfer_server:app", host="0.0.0.0", port=8000, reload=True) 