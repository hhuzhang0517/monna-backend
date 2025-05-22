import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List, Callable, Awaitable
import redis.asyncio as aioredis

from app.core.config import settings
from app.models.schemas import TaskStatus
from app.worker.queue import add_task, get_task_status, update_task_status

# 配置日志
logger = logging.getLogger(__name__)

# 创建Redis专用日志处理器
os.makedirs(settings.LOGS_DIR, exist_ok=True)
redis_log_file = os.path.join(settings.LOGS_DIR, "redis_queue.log")
file_handler = logging.FileHandler(redis_log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)  # 设置为DEBUG级别，记录更多详细信息

# Redis连接
redis_client = None
subscriber = None
redis_running = False

# Redis配置
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# 队列和通道名称
TASK_QUEUE = "ai_tasks"  # 与Node.js API使用的队列名称保持一致
TASK_STATUS_CHANNEL = "task_status"
PHOTO_TASK_PATTERN = "photo_task:*"

async def initialize_redis():
    """初始化Redis连接"""
    global redis_client, subscriber, redis_running
    try:
        # 创建Redis连接
        connection_url = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        if REDIS_PASSWORD:
            connection_url = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
        
        redis_client = aioredis.from_url(connection_url, decode_responses=True)
        subscriber = redis_client.pubsub()
        
        # 测试连接
        await redis_client.ping()
        logger.info(f"成功连接到Redis服务器: {REDIS_HOST}:{REDIS_PORT}")
        
        # 订阅任务状态通道
        await subscriber.subscribe(TASK_STATUS_CHANNEL)
        logger.info(f"已订阅Redis通道: {TASK_STATUS_CHANNEL}")
        
        # 启动监听器
        redis_running = True
        asyncio.create_task(listen_for_messages())
        asyncio.create_task(poll_task_queue())
        
        return True
    except Exception as e:
        logger.error(f"连接Redis服务器失败: {str(e)}")
        return False

async def listen_for_messages():
    """监听Redis通道的消息"""
    global subscriber, redis_running
    logger.info("开始监听Redis消息通道")
    
    try:
        while redis_running and subscriber:
            message = await subscriber.get_message(ignore_subscribe_messages=True)
            if message:
                try:
                    channel = message["channel"]
                    data = json.loads(message["data"])
                    logger.info(f"收到Redis消息: channel={channel}, data={data}")
                    
                    # 处理任务状态更新
                    if channel == TASK_STATUS_CHANNEL and "task_id" in data:
                        task_id = data["task_id"]
                        status = data.get("status")
                        progress = data.get("progress")
                        logger.info(f"任务状态更新: {task_id}, 状态: {status}, 进度: {progress}")
                except Exception as e:
                    logger.error(f"处理Redis消息时出错: {str(e)}")
            
            # 短暂休眠，避免CPU使用率过高
            await asyncio.sleep(0.1)
    except Exception as e:
        logger.exception(f"Redis消息监听器异常: {str(e)}")
    finally:
        logger.info("Redis消息监听器已停止")

async def poll_task_queue():
    """轮询Redis任务队列"""
    global redis_client, redis_running
    logger.info(f"开始轮询Redis任务队列: {TASK_QUEUE}")
    
    try:
        while redis_running and redis_client:
            # 使用LPOP从队列中取出任务
            try:
                # 检查队列长度
                queue_length = await redis_client.llen(TASK_QUEUE)
                if queue_length > 0:
                    logger.info(f"Redis队列 {TASK_QUEUE} 中有 {queue_length} 个任务待处理")
                
                task_data_str = await redis_client.lpop(TASK_QUEUE)
                if task_data_str:
                    # 解析任务数据
                    task_data = json.loads(task_data_str)
                    task_id = task_data.get("task_id")
                    task_type = task_data.get("type", "unknown")
                    
                    logger.info(f"从Redis队列获取到任务: {task_id}, 类型: {task_type}, 数据: {task_data}")
                    
                    # 如果是写真任务，转发到FaceChain处理
                    if task_type == "photo" and task_id:
                        await process_photo_task(task_id, task_data)
                    else:
                        logger.warning(f"未知任务类型: {task_type}, 任务ID: {task_id}")
            except Exception as e:
                logger.error(f"处理Redis任务时出错: {str(e)}", exc_info=True)
            
            # 等待一段时间再检查队列
            await asyncio.sleep(1)
    except Exception as e:
        logger.exception(f"Redis任务队列轮询器异常: {str(e)}")
    finally:
        logger.info("Redis任务队列轮询器已停止")

async def process_photo_task(task_id: str, task_data: Dict[str, Any]):
    """处理来自Redis的写真任务"""
    try:
        logger.info(f"处理Redis写真任务: {task_id}")
        
        # 转换为FaceChain任务格式
        input_files = []
        style = task_data.get("style", "标准")
        count = task_data.get("count", 5)
        
        # 获取上传的照片路径
        upload_dir = os.path.join(settings.BASE_DIR, "data", "uploads", task_id)
        output_dir = os.path.join(settings.BASE_DIR, "data", "outputs", task_id)
        
        if os.path.exists(upload_dir):
            for file in os.listdir(upload_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input_files.append(os.path.join(upload_dir, file))
        
        if not input_files:
            logger.error(f"任务 {task_id} 没有找到有效的输入文件")
            await update_task_status_redis(task_id, "failed", 0, "没有找到有效的输入文件")
            return
        
        # 准备FaceChain任务数据
        facechain_task = {
            "task_id": task_id,
            "input_files": input_files,
            "output_dir": output_dir,
            "style": style,
            "num_generate": count,
            "multiplier_style": 0.25,  # 默认风格强度
            "use_pose": False,
            "pose_file": None,
        }
        
        # 尝试使用Celery任务处理
        try:
            # 首先尝试使用Celery处理任务
            await update_task_status_redis(task_id, "processing", 5, "正在准备Celery任务处理")
            success = await submit_to_celery(task_id, facechain_task)
            if success:
                logger.info(f"已将Redis任务 {task_id} 提交到Celery处理")
                return
            logger.warning(f"Celery提交失败，尝试使用内存队列处理任务 {task_id}")
        except Exception as e:
            logger.exception(f"Celery提交失败: {str(e)}，尝试使用内存队列")
        
        # 作为备选方案，添加到内存队列进行处理
        add_task(task_id, facechain_task)
        logger.info(f"已将Redis任务 {task_id} 添加到FaceChain内存队列")
        
        # 更新Redis中的任务状态
        await update_task_status_redis(task_id, "processing", 10, "已加入FaceChain处理队列")
    except Exception as e:
        logger.exception(f"处理Redis写真任务 {task_id} 失败: {str(e)}")
        await update_task_status_redis(task_id, "failed", 0, f"处理失败: {str(e)}")

async def submit_to_celery(task_id: str, task_data: Dict[str, Any]) -> bool:
    """将任务提交到Celery处理"""
    try:
        # 导入Celery任务
        from app.worker.tasks import process_facechain_portrait
        
        # 添加详细日志
        logger.info(f"准备提交任务 {task_id} 到Celery处理")
        logger.info(f"任务数据: {json.dumps(task_data, ensure_ascii=False)}")
        
        # 调用Celery任务（异步）
        celery_task = process_facechain_portrait.delay(task_data)
        
        logger.info(f"Celery任务已提交，任务ID: {celery_task.id}, Celery任务状态: {celery_task.state}")
        
        # 等待一会儿并检查任务状态
        await asyncio.sleep(1)
        
        try:
            # 检查任务状态
            task_state = celery_task.state
            logger.info(f"Celery任务 {celery_task.id} 的状态: {task_state}")
            
            if task_state == 'PENDING' or task_state == 'RECEIVED' or task_state == 'STARTED':
                logger.info(f"Celery任务 {celery_task.id} 已被接收处理")
                return True
            else:
                logger.warning(f"Celery任务 {celery_task.id} 状态异常: {task_state}")
                return False
        except Exception as e:
            logger.warning(f"检查Celery任务状态时出错: {str(e)}")
            # 仍然返回True，因为任务已提交
            return True
        
        return True
    except ImportError as e:
        logger.error(f"Celery任务模块导入失败: {str(e)}")
        return False
    except Exception as e:
        logger.exception(f"提交到Celery失败: {str(e)}")
        return False

async def update_task_status_redis(task_id: str, status: str, progress: int, message: str = ""):
    """更新Redis中的任务状态"""
    global redis_client
    try:
        if redis_client:
            status_data = {
                "task_id": task_id,
                "status": status,
                "progress": progress,
                "message": message,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # 发布状态更新到通道
            channel = f"task_status:{task_id}"
            await redis_client.publish(channel, json.dumps(status_data))
            
            # 同时更新任务状态哈希表
            status_key = f"task:{task_id}"
            await redis_client.hset(status_key, mapping={
                "status": status,
                "progress": progress,
                "message": message,
                "updated_at": asyncio.get_event_loop().time()
            })
            
            logger.info(f"已更新Redis中任务 {task_id} 的状态: {status}, 进度: {progress}")
    except Exception as e:
        logger.error(f"更新Redis任务状态失败: {str(e)}")

async def shutdown_redis():
    """关闭Redis连接"""
    global redis_client, subscriber, redis_running
    redis_running = False
    
    try:
        if subscriber:
            await subscriber.unsubscribe()
            await subscriber.close()
        
        if redis_client:
            await redis_client.close()
            
        logger.info("Redis连接已关闭")
    except Exception as e:
        logger.error(f"关闭Redis连接时出错: {str(e)}") 