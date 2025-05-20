#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Monna AI 照片生成工作器
连接Redis队列，处理照片生成任务
"""

import os
import sys
import json
import logging
import signal
import time
import uuid
import redis
from pathlib import Path
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('PhotoWorker')

# 导入FaceChain生成器
try:
    from facechain_integration import facechain_generator
    logger.info("FaceChain模块已成功加载")
except ImportError as e:
    logger.error(f"无法加载FaceChain模块: {e}")
    sys.exit(1)
    
# 导入样本生成器
try:
    # 仅当需要时导入
    import importlib.util
    sample_generator_path = Path(__file__).parent / "generate_sample.py"
    if sample_generator_path.exists():
        logger.info("找到样本生成器模块")
    else:
        logger.warning("未找到样本生成器模块，不能使用备用生成方法")
except ImportError as e:
    logger.warning(f"无法加载样本生成器: {e}")

# 配置Redis连接
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
REDIS_DB = int(os.environ.get('REDIS_DB', 0))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)

# 任务队列名称
TASK_QUEUE = 'ai_tasks'
TASK_STATUS_CHANNEL = 'task_status'

class PhotoWorker:
    def __init__(self):
        """初始化照片生成工作器"""
        self.redis_client = None
        self.running = True
        self.connect_redis()
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        
        # 基础路径
        self.base_dir = Path(__file__).parent.parent
        self.uploads_dir = self.base_dir / 'data' / 'uploads'
        self.outputs_dir = self.base_dir / 'data' / 'outputs'
        self.models_dir = self.base_dir / 'models'
        
        logger.info(f"工作目录: {self.base_dir}")
        logger.info(f"模型目录: {self.models_dir}")
        logger.info(f"上传目录: {self.uploads_dir}")
        logger.info(f"输出目录: {self.outputs_dir}")
    
    def connect_redis(self):
        """连接Redis服务器"""
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                decode_responses=True  # 自动解码为字符串
            )
            self.redis_client.ping()  # 测试连接
            logger.info(f"成功连接到Redis: {REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}")
        except redis.RedisError as e:
            logger.error(f"连接Redis失败: {e}")
            sys.exit(1)
    
    def handle_signal(self, signum, frame):
        """处理信号"""
        logger.info(f"收到终止信号，停止工作器")
        self.running = False
    
    def publish_status(self, task_id, status, progress=None, results=None):
        """发布任务状态更新"""
        try:
            # 构建状态信息
            status_data = {
                'taskId': task_id,
                'status': status,
                'updatedAt': time.strftime('%Y-%m-%dT%H:%M:%S')
            }
            
            if progress is not None:
                status_data['progress'] = progress
                
            if results is not None:
                status_data['results'] = results
            
            # 发布到Redis
            self.redis_client.publish(
                TASK_STATUS_CHANNEL,
                json.dumps(status_data)
            )
            logger.info(f"已发布任务 {task_id} 状态更新: {status}")
            return True
        except Exception as e:
            logger.error(f"发布状态更新失败: {e}")
            return False
    
    def generate_photos(self, task_id, photos, styles, output_dir):
        """生成照片，首先尝试FaceChain，失败则使用样本生成器"""
        try:
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 首先尝试使用FaceChain
            if facechain_generator.initialized:
                logger.info(f"使用FaceChain生成照片...")
                result = facechain_generator.generate(
                    task_id,
                    photos,
                    styles,
                    output_dir
                )
                
                if result:
                    logger.info(f"FaceChain生成照片成功")
                    return True
                else:
                    logger.warning(f"FaceChain生成照片失败，尝试使用样本生成器")
            else:
                logger.warning(f"FaceChain未初始化，尝试使用样本生成器")
                
            # 如果FaceChain失败，尝试使用样本生成器
            try:
                # 动态导入样本生成器模块
                sample_generator_path = Path(__file__).parent / "generate_sample.py"
                if not sample_generator_path.exists():
                    logger.error("样本生成器模块不存在")
                    return False
                
                spec = importlib.util.spec_from_file_location("sample_generator", sample_generator_path)
                sample_generator = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(sample_generator)
                
                # 调用样本生成器
                logger.info(f"使用样本生成器生成照片...")
                sample_generator.generate_samples(photos, styles, output_dir)
                logger.info(f"样本生成器生成照片成功")
                return True
            except Exception as e:
                logger.error(f"样本生成器失败: {e}")
                logger.error(traceback.format_exc())
                return False
        except Exception as e:
            logger.error(f"生成照片时发生错误: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def process_task(self, task_data):
        """处理单个任务"""
        try:
            # 解析任务数据
            if isinstance(task_data, str):
                task_data = json.loads(task_data)
                
            # 提取任务信息
            task_id = task_data.get('taskId')
            task_type = task_data.get('taskType', 'photo')
            file_paths = task_data.get('filePaths', [])
            styles = task_data.get('styles', ['portrait'])
            
            if not task_id:
                logger.error("任务缺少taskId")
                return False
                
            if not file_paths:
                logger.error(f"任务 {task_id} 没有指定照片路径")
                return False
            
            logger.info(f"开始处理任务 {task_id}")
            logger.info(f"处理 {len(file_paths)} 张照片，风格: {styles}")
            
            # 更新状态为处理中
            self.publish_status(task_id, 'processing')
            
            # 准备输出目录
            output_dir = self.outputs_dir / task_id
            
            # 生成照片
            success = self.generate_photos(
                task_id,
                file_paths,
                styles,
                output_dir
            )
            
            if success:
                # 获取结果文件列表
                result_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    result_files.extend(list(output_dir.glob(f'**/{ext}')))
                
                # 更新任务状态为完成
                self.publish_status(
                    task_id, 
                    'finished',
                    results={
                        'files': [str(f.relative_to(self.base_dir)) for f in result_files]
                    }
                )
                
                logger.info(f"任务 {task_id} 完成，生成了 {len(result_files)} 个 文件")
                return True
            else:
                # 更新任务状态为失败
                self.publish_status(task_id, 'failed')
                logger.error(f"任务 {task_id} 失败")
                return False
                
        except Exception as e:
            logger.error(f"处理任务时发生错误: {e}")
            logger.error(traceback.format_exc())
            
            # 尝试发布失败状态
            try:
                if task_id:
                    self.publish_status(task_id, 'failed')
            except:
                pass
                
            return False
    
    def run(self):
        """启动工作器主循环"""
        logger.info(f"启动AI照片生成工作器，监听任务队列...")
        
        while self.running:
            try:
                # 从队列中阻塞获取任务
                task = self.redis_client.blpop(TASK_QUEUE, timeout=1)
                
                if not task:
                    # 超时，继续循环
                    continue
                    
                # task是一个元组 (queue_name, task_data)
                _, task_data = task
                
                # 解析JSON任务数据
                try:
                    if isinstance(task_data, bytes):
                        task_data = task_data.decode('utf-8')
                        
                    task_json = json.loads(task_data)
                    logger.info(f"收到新任务: {task_json.get('taskId')}")
                    
                    # 处理任务
                    self.process_task(task_json)
                except json.JSONDecodeError:
                    logger.error(f"无效的任务数据格式: {task_data}")
                    continue
                    
            except redis.RedisError as e:
                logger.error(f"Redis错误: {e}")
                time.sleep(5)  # 避免因连接问题而过度消耗资源
                
                # 尝试重新连接
                try:
                    self.connect_redis()
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"工作器处理任务异常: {e}")
                logger.error(traceback.format_exc())
                time.sleep(1)  # 避免因错误而过度消耗资源
        
        logger.info("工作器已停止")

def main():
    """主函数"""
    worker = PhotoWorker()
    worker.run()

if __name__ == "__main__":
    main() 