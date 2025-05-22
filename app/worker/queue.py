import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Awaitable
from collections import deque
import threading

from app.core.config import settings
from app.models.schemas import TaskStatus

# 配置日志
logger = logging.getLogger(__name__)

# 任务队列
task_queue = deque()
queue_lock = threading.Lock()
worker_running = False
processor_running = False

# 任务状态存储
tasks_status: Dict[str, Dict[str, Any]] = {}

async def initialize_queue():
    """初始化任务队列系统"""
    logger.info("正在初始化任务队列系统...")
    # 启动任务处理器，如果没有在运行
    global processor_running
    if not processor_running:
        processor_running = True
        asyncio.create_task(process_queue())
    return True

def add_task(task_id: str, task_data: Dict[str, Any]):
    """
    添加任务到队列
    
    Args:
        task_id: 任务唯一标识
        task_data: 任务数据，包含处理所需的所有信息
    """
    with queue_lock:
        tasks_status[task_id] = {
            "status": TaskStatus.PENDING,
            "progress": 0,
            "created_at": time.time(),
            **task_data
        }
        task_queue.append(task_id)
        logger.info(f"任务 {task_id} 已添加到队列，当前队列长度: {len(task_queue)}")

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    """获取任务状态"""
    return tasks_status.get(task_id)

def update_task_status(task_id: str, status: TaskStatus, progress: int = None, result: Any = None):
    """更新任务状态"""
    if task_id in tasks_status:
        tasks_status[task_id]["status"] = status
        if progress is not None:
            tasks_status[task_id]["progress"] = progress
        if result is not None:
            tasks_status[task_id]["result"] = result
        logger.info(f"任务 {task_id} 状态更新: {status}, 进度: {progress}")

async def process_task(task_id: str, processor: Callable[[str, Dict[str, Any]], Awaitable[Any]]):
    """处理单个任务"""
    if task_id not in tasks_status:
        logger.error(f"任务 {task_id} 不存在")
        return
        
    task_data = tasks_status[task_id]
    try:
        # 更新任务状态为处理中
        update_task_status(task_id, TaskStatus.PROCESSING, 0)
        
        # 调用处理器处理任务
        result = await processor(task_id, task_data)
        
        # 更新任务状态为完成
        update_task_status(task_id, TaskStatus.COMPLETED, 100, result)
        
        logger.info(f"任务 {task_id} 处理完成")
        return result
    except Exception as e:
        logger.exception(f"处理任务 {task_id} 时出错: {str(e)}")
        update_task_status(task_id, TaskStatus.FAILED, progress=0)
        return None

async def process_queue():
    """持续处理队列中的任务"""
    global processor_running
    logger.info("任务队列处理器已启动")
    
    try:
        while processor_running:
            # 检查队列中是否有任务
            task_id = None
            with queue_lock:
                if task_queue:
                    task_id = task_queue.popleft()
            
            if task_id:
                logger.info(f"开始处理任务 {task_id}")
                from app.api.endpoints.facechain import process_facechain_task
                
                # 处理任务
                try:
                    await process_facechain_task(task_id, "从队列处理器调用")
                except Exception as e:
                    logger.exception(f"处理任务 {task_id} 时出错: {str(e)}")
                    update_task_status(task_id, TaskStatus.FAILED)
            
            # 短暂休眠，避免CPU占用过高
            await asyncio.sleep(1)
    except Exception as e:
        logger.exception(f"任务队列处理器出错: {str(e)}")
    finally:
        processor_running = False
        logger.info("任务队列处理器已停止")

def shutdown_queue():
    """关闭任务队列系统"""
    global processor_running
    processor_running = False
    logger.info("任务队列系统已关闭") 