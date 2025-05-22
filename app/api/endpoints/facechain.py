from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks, Form, Query, Path, Request
from typing import Optional, Dict, Any, List
import json
import os
import time
import uuid
import logging
import subprocess
import sys
from pathlib import Path as PathLib
import shutil
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np
import asyncio

from app.models.schemas import ImageUploadResponse, TaskStatus, TaskStatusResponse
from app.core.config import settings

# FaceChain specific imports - 确保这些路径相对于models/facechain目录或者已在PYTHONPATH中
# 我们会在调用时切换工作目录
from facechain.inference_fact import GenPortrait as FaceChainGenPortrait
from facechain.utils import snapshot_download
from facechain.constants import neg_prompt, pos_prompt_with_cloth, pos_prompt_with_style, base_models
import cv2

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(settings.BASE_DIR / "logs" / "facechain_api.log")),
        logging.StreamHandler()
    ]
)

# 确保日志目录存在
os.makedirs(settings.BASE_DIR / "logs", exist_ok=True)

# 任务状态枚举
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 响应模型 - 使用Pydantic BaseModel
class ImageUploadResponse(BaseModel):
    task_id: str
    status: TaskStatus 
    message: str
    
    class Config:
        # 允许使用额外的字段
        extra = "allow"

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int = 0
    created_at: float
    style: str
    count: int
    result_urls: List[str] = []
    message: str = ""
    apiBaseUrl: str = ""
    pythonApiBaseUrl: str = ""
    
    class Config:
        # 允许使用额外的字段
        extra = "allow"

router = APIRouter()

# 确保输出目录存在
os.makedirs(settings.BASE_DIR / "data" / "outputs", exist_ok=True)

# 简单的任务状态存储
facechain_tasks: Dict[str, Dict[str, Any]] = {}

# --- 全局模型实例和风格数据 ---
gen_portrait_instance: Optional[FaceChainGenPortrait] = None
loaded_styles: Optional[List[Dict]] = None
facechain_models_dir = settings.BASE_DIR / "models" / "facechain"

def get_facechain_styles():
    global loaded_styles
    if loaded_styles is None:
        logger.info(f"Loading FaceChain styles from {facechain_models_dir / 'styles'}...")
        styles_data = []
        original_cwd = os.getcwd()
        try:
            os.chdir(facechain_models_dir)
            for base_model_conf in base_models: # base_models来自facechain.constants
                style_in_base = []
                folder_path = os.path.join("styles", base_model_conf['name'])
                if not os.path.isdir(folder_path):
                    logger.warning(f"Style folder not found: {folder_path} in {os.getcwd()}")
                    continue
                files = os.listdir(folder_path)
                files.sort()
                for file in files:
                    file_path = os.path.join(folder_path, file)
                    try:
                        with open(file_path, "r", encoding='utf-8') as f:
                            data = json.load(f)
                            style_in_base.append(data['name'])
                            styles_data.append(data)
                    except Exception as e_json:
                        logger.error(f"Error loading style JSON {file_path}: {e_json}")
                base_model_conf['style_list'] = style_in_base # 更新原始constants中的base_models字典
            loaded_styles = styles_data
            logger.info(f"Successfully loaded {len(loaded_styles)} FaceChain styles.")
        except Exception as e_load_styles:
            logger.exception(f"Failed to load FaceChain styles: {e_load_styles}")
        finally:
            os.chdir(original_cwd)
    return loaded_styles

def initialize_facechain_model():
    global gen_portrait_instance
    if gen_portrait_instance is None:
        logger.info("Initializing FaceChain GenPortrait model...")
        original_cwd = os.getcwd()
        try:
            os.chdir(facechain_models_dir)
            # 确保GenPortrait在此工作目录下能正确初始化模型和依赖
            gen_portrait_instance = FaceChainGenPortrait()
            logger.info("FaceChain GenPortrait model initialized successfully.")
        except Exception as e_init_model:
            logger.exception(f"Failed to initialize FaceChain GenPortrait model: {e_init_model}")
        finally:
            os.chdir(original_cwd)
    get_facechain_styles() # 加载风格，确保在模型初始化后或可独立加载

@router.get("/connection-test", summary="连接测试端点")
async def test_connection(request: Request):
    """
    用于前端测试FaceChain服务连接状态的端点
    
    返回一个简单的成功消息，添加apiBaseUrl字段修复前端startsWith错误
    """
    logger.info("接收到连接测试请求")
    
    # 构建API基础URL
    host = request.headers.get("host", "localhost:8000")
    scheme = request.headers.get("x-forwarded-proto", "http")
    api_base_url = f"{scheme}://{host}/api/v1"
    
    return {
        "status": "ok",
        "message": "FaceChain服务连接正常",
        "service": "facechain",
        "timestamp": time.time(),
        "apiBaseUrl": api_base_url,  # 添加此字段以修复前端startsWith错误
        "pythonApiBaseUrl": api_base_url,  # 添加可能被前端使用的字段
        "nodeApiBaseUrl": f"{scheme}://{host.split(':')[0]}:3001/api"  # Node API基础URL
    }

@router.post("/generate-portrait", summary="AI人像生成", response_model=ImageUploadResponse)
async def generate_portrait(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="上传4-12张人像照片"),
    style: str = Form(..., description="生成风格, 可选值参考facechain的styles目录"),
    num_generate: int = Form(5, description="生成数量，默认为5张"),
    multiplier_style: float = Form(0.25, description="风格强度系数，范围0-1"),
    use_pose: bool = Form(False, description="是否使用指定姿势"),
    pose_image: Optional[UploadFile] = File(None, description="姿势参考图像(可选)")
):
    """
    使用FaceChain AI模型生成人像照片
    
    - **files**: 上传4-12张人像照片，这些照片将用于训练模型
    - **style**: 生成风格，对应facechain的styles目录下的风格
    - **num_generate**: 要生成的图像数量
    - **multiplier_style**: 风格强度系数，越大风格特征越明显，范围0-1
    - **use_pose**: 是否使用指定姿势
    - **pose_image**: 姿势参考图像(仅在use_pose=True时有效)
    """
    # 验证文件数量
    if len(files) < 4 or len(files) > 12:
        raise HTTPException(
            status_code=400, 
            detail=f"请上传4-12张照片，当前上传了{len(files)}张"
        )
    
    # 验证文件类型
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
            raise HTTPException(
                status_code=400, 
                detail=f"文件 {file.filename} 不是支持的图片格式，请使用JPG或PNG格式"
            )
    
    # 验证生成数量范围
    if num_generate < 1 or num_generate > 20:
        raise HTTPException(
            status_code=400,
            detail="生成数量必须在1-20之间"
        )
    
    # 验证风格强度系数范围
    if not (0 <= multiplier_style <= 1):
        raise HTTPException(
            status_code=400,
            detail="风格强度系数必须在0-1范围内"
        )
    
    # 生成唯一任务ID
    task_id = str(uuid.uuid4())
    logger.info(f"接收到AI人像生成请求，创建任务 {task_id} for style '{style}'")
    
    # 创建任务目录
    task_dir = settings.UPLOAD_DIR / "facechain" / task_id
    output_dir = settings.BASE_DIR / "data" / "outputs" / task_id
    
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存上传的图像
    input_files = []
    for i, file in enumerate(files):
        # 生成文件名
        filename = f"input_{i}{PathLib(file.filename).suffix}"
        file_path = task_dir / filename
        
        # 保存文件
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        input_files.append(str(file_path))
    
    # 保存姿势图像(如果有)
    pose_file_path = None
    if use_pose and pose_image:
        pose_filename = f"pose{PathLib(pose_image.filename).suffix}"
        pose_file_path = task_dir / pose_filename
        
        with open(pose_file_path, "wb") as buffer:
            shutil.copyfileobj(pose_image.file, buffer)
    
    # 初始化任务状态
    facechain_tasks[task_id] = {
        "status": TaskStatus.PENDING,
        "progress": 0,
        "created_at": time.time(),
        "input_files": input_files,
        "output_dir": str(output_dir),
        "style": style,
        "num_generate": num_generate,
        "multiplier_style": multiplier_style,
        "use_pose": use_pose,
        "pose_file": str(pose_file_path) if pose_file_path else None,
        "results": []
    }
    
    # 使用队列系统处理任务
    from app.worker.queue import add_task
    
    # 准备任务数据
    task_data = {
        "input_files": input_files,
        "output_dir": str(output_dir),
        "style": style,
        "count": num_generate,
        "multiplier_style": multiplier_style,
        "use_pose": use_pose,
        "pose_file": str(pose_file_path) if pose_file_path else None,
    }
    
    # 添加到任务队列
    add_task(task_id, task_data)
    logger.info(f"任务 {task_id} 已添加到队列，等待处理")
    
    return ImageUploadResponse(
        task_id=task_id,
        status=TaskStatus.PENDING,
        message="AI人像生成任务已提交，正在处理中"
    )

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str = Path(..., description="任务ID"), request: Request = None):
    """
    获取任务状态和结果
    
    - **task_id**: 任务ID
    """
    # 构建API基础URL (无论任务是否存在，都需要确保此字段存在以避免前端startsWith错误)
    host = request.headers.get("host", "localhost:8000")
    scheme = request.headers.get("x-forwarded-proto", "http")
    api_base_url = f"{scheme}://{host}/api/v1"
    
    # 测试ID用于连接测试
    if task_id == "test-id":
        return TaskStatusResponse(
            task_id=task_id,
            status=TaskStatus.PENDING,
            progress=0,
            created_at=time.time(),
            style="",
            count=0,
            message="测试连接成功",
            apiBaseUrl=api_base_url,  # 添加此字段以修复前端startsWith错误
            pythonApiBaseUrl=api_base_url  # 为前端提供Python API的基础URL
        )
    
    # 从队列系统获取任务状态
    from app.worker.queue import get_task_status
    
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 将结果文件路径转换为URL
    result_urls = []
    if task.get("status") == TaskStatus.COMPLETED and task.get("result"):
        for result_file in task["result"]:
            # 提取相对路径
            rel_path = os.path.relpath(result_file, str(settings.BASE_DIR))
            # 构造URL
            url = f"/api/v1/static/{rel_path.replace(os.sep, '/')}"
            result_urls.append(url)
    
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        created_at=task["created_at"],
        style=task.get("style", ""),
        count=task.get("count", 0),
        result_urls=result_urls,
        message=task.get("message", ""),
        apiBaseUrl=api_base_url,
        pythonApiBaseUrl=api_base_url
    )

@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str = Path(..., description="任务ID")):
    """
    删除任务和相关文件
    
    - **task_id**: 任务ID
    """
    # 从队列系统获取任务状态
    from app.worker.queue import get_task_status, tasks_status
    
    task = get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 删除输入文件目录
    input_dir = settings.UPLOAD_DIR / "facechain" / task_id
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    
    # 删除输出文件目录
    output_dir = PathLib(task.get("output_dir", ""))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # 删除任务记录
    if task_id in tasks_status:
        del tasks_status[task_id]
    
    return {"message": f"任务 {task_id} 及相关文件已删除"}

# 后台处理函数
async def process_facechain_task(
    task_id: str,
    message_for_debug: str = "默认调用"
):
    """处理FaceChain生成任务"""
    # 获取任务数据
    from app.worker.queue import get_task_status, update_task_status
    
    # --- 调试日志记录 ---
    debug_signal_file = settings.BASE_DIR / "logs" / "debug_task_signal.txt"
    try:
        os.makedirs(settings.BASE_DIR / "logs", exist_ok=True)
        with open(debug_signal_file, "a", encoding='utf-8') as f:
            f.write(f"{time.time()}: process_facechain_task CALLED for {task_id} with message: {message_for_debug}\n")
    except Exception as e_debug_write:
        sys.stdout.write(f"CRITICAL DEBUG: Failed to write to {debug_signal_file}: {e_debug_write}\n")
        sys.stdout.flush()

    logger.info(f"--- DEBUG LOGGER --- process_facechain_task for task_id '{task_id}' CALLED with message: '{message_for_debug}' ---")
    
    # 获取任务数据
    task_data = get_task_status(task_id)
    if not task_data:
        logger.error(f"任务 {task_id} 不存在于队列系统中")
        return
    
    try:
        # 更新任务状态为处理中
        update_task_status(task_id, TaskStatus.PROCESSING, 10)
        logger.info(f"开始处理FaceChain任务 {task_id} for style '{task_data.get('style', 'unknown')}'")

        if gen_portrait_instance is None or loaded_styles is None:
            logger.error(f"Task {task_id}: FaceChain model or styles not initialized. Aborting task.")
            update_task_status(task_id, TaskStatus.FAILED, progress=100)
            return

        try:
            # 更新进度
            update_task_status(task_id, TaskStatus.PROCESSING, 20)

            # 获取任务参数
            input_files = task_data.get('input_files', [])
            output_dir = task_data.get('output_dir')
            style = task_data.get('style')
            num_generate = task_data.get('count', 5)
            multiplier_style = task_data.get('multiplier_style', 0.25)
            use_pose = task_data.get('use_pose', False)
            pose_file = task_data.get('pose_file')

            if not input_files:
                raise ValueError("没有提供输入图像文件")
            
            primary_input_path = input_files[0]

            # 查找目标风格配置
            target_style_config = None
            for s_config in loaded_styles:
                if s_config['name'] == style:
                    target_style_config = s_config
                    break
            
            if not target_style_config:
                logger.error(f"错误: 未找到指定的风格 '{style}' for task {task_id}")
                update_task_status(task_id, TaskStatus.FAILED, 100)
                return

            update_task_status(task_id, TaskStatus.PROCESSING, 30)
            
            # 准备推理参数
            model_id = target_style_config['model_id']
            base_model_idx = 0 
            
            current_cwd = os.getcwd()
            style_model_path_for_inference = None
            pos_prompt_for_inference = ""
            
            try:
                # 切换到FaceChain模型目录进行处理
                os.chdir(facechain_models_dir) 
                if model_id is None:
                    style_model_path_for_inference = None
                    pos_prompt_for_inference = pos_prompt_with_cloth.format(target_style_config['add_prompt_style'])
                else:
                    model_dir_local = model_id if os.path.exists(model_id) else snapshot_download(model_id, revision=target_style_config['revision'])
                    style_model_path_for_inference = os.path.join(model_dir_local, target_style_config['bin_file'])
                    pos_prompt_for_inference = pos_prompt_with_style.format(target_style_config['add_prompt_style'])
            finally:
                os.chdir(current_cwd)

            # 处理姿势图像
            pose_image_for_inference = None
            if use_pose and pose_file and os.path.exists(pose_file):
                pose_image_for_inference = pose_file
            
            logger.info(f"Task {task_id} FaceChain inference params: num_generate={num_generate}, base_model_idx={base_model_idx}, "
                      f"style_model_path='{style_model_path_for_inference}', pos_prompt='{pos_prompt_for_inference[:60]}...', "
                      f"input_img_path='{primary_input_path}', pose_image='{pose_image_for_inference}', "
                      f"multiplier_style={multiplier_style}")

            update_task_status(task_id, TaskStatus.PROCESSING, 40)
            outputs_np_arrays = []
            task_specific_output_dir = PathLib(output_dir)

            try:
                # 执行FaceChain推理
                logger.info(f"Task {task_id}: Changing CWD to {facechain_models_dir} for inference.")
                os.chdir(facechain_models_dir) 
                outputs_np_arrays = gen_portrait_instance(
                    num_images=num_generate,
                    base_model_index=base_model_idx,
                    style_model_path=style_model_path_for_inference,
                    pos_prompt=pos_prompt_for_inference,
                    neg_prompt=neg_prompt, 
                    input_image_path=primary_input_path, 
                    pose_image_path=pose_image_for_inference, 
                    multiplier_style=multiplier_style
                )
                logger.info(f"Task {task_id}: Inference call completed in CWD: {os.getcwd()}")
            finally:
                os.chdir(current_cwd) 
                logger.info(f"Task {task_id}: Restored CWD to {current_cwd}")

            update_task_status(task_id, TaskStatus.PROCESSING, 80)
            logger.info(f"Task {task_id}: FaceChain inference completed, got {len(outputs_np_arrays)} image arrays.")

            # 保存生成的图像
            result_image_paths = []
            if not outputs_np_arrays:
                logger.warning(f"Task {task_id}: FaceChain inference returned no image arrays.")

            for i, out_np_array in enumerate(outputs_np_arrays):
                output_image_filename = f"result_{i}.png"
                output_image_filepath = task_specific_output_dir / output_image_filename
                try:
                    if not isinstance(out_np_array, np.ndarray):
                        logger.error(f"Task {task_id}: Output {i} is not a numpy array, type: {type(out_np_array)}. Skipping save.")
                        continue
                    cv2.imwrite(str(output_image_filepath), out_np_array)
                    result_image_paths.append(str(output_image_filepath))
                    logger.info(f"Task {task_id}: Saved generated image to {output_image_filepath}")
                except Exception as e_save:
                    logger.error(f"Task {task_id}: Error saving image {output_image_filepath}: {e_save}")
            
            if not result_image_paths and outputs_np_arrays:
                 logger.error(f"Task {task_id}: Images were generated but failed to save.")
                 raise Exception("图像已生成但保存失败")
            elif not result_image_paths: 
                logger.error(f"Task {task_id}: No images were generated by FaceChain.")
                raise Exception("FaceChain未能生成任何图像")

            # 任务完成，更新状态
            update_task_status(task_id, TaskStatus.COMPLETED, 100, result=result_image_paths)
            logger.info(f"Task {task_id} completed successfully.")

        except Exception as e_inner_task: 
            logger.exception(f"处理FaceChain任务 {task_id} 内部发生错误: {e_inner_task}")
            update_task_status(task_id, TaskStatus.FAILED, 100)
    
    except Exception as e_outer_task: 
        logger.exception(f"FaceChain任务 {task_id} 顶层处理中发生严重错误: {e_outer_task}")
        update_task_status(task_id, TaskStatus.FAILED, 100) 