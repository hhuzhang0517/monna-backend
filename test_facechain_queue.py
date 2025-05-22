import os
import sys
import asyncio
import logging
from pathlib import Path
import time
import uuid
import shutil

# 设置项目根目录
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(project_root / "logs" / "test_queue.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("facechain_queue_test")

# 导入必要的模块
from app.core.config import settings
from app.models.schemas import TaskStatus
from app.api.endpoints.facechain import initialize_facechain_model, process_facechain_task
from app.worker.queue import initialize_queue, add_task, get_task_status, update_task_status

async def test_facechain_queue():
    """测试FaceChain任务队列处理流程"""
    logger.info("开始测试FaceChain任务队列处理流程")
    
    # 初始化FaceChain模型
    logger.info("初始化FaceChain模型...")
    initialize_facechain_model()
    
    # 初始化任务队列
    logger.info("初始化任务队列...")
    await initialize_queue()
    
    # 创建测试目录
    task_id = str(uuid.uuid4())
    logger.info(f"创建测试任务: {task_id}")
    
    task_dir = settings.UPLOAD_DIR / "facechain" / task_id
    output_dir = settings.BASE_DIR / "data" / "outputs" / task_id
    
    os.makedirs(task_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制测试图片
    test_image_source = None
    for test_dir in ["test-images", "data/samples"]:
        test_dir_path = project_root / test_dir
        if test_dir_path.exists():
            for file in os.listdir(test_dir_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    test_image_source = test_dir_path / file
                    break
            if test_image_source:
                break
    
    if not test_image_source:
        logger.error("未找到测试图片，请在test-images或data/samples目录中放置图片")
        return False
    
    input_file = task_dir / f"input_0{Path(test_image_source).suffix}"
    shutil.copy(test_image_source, input_file)
    logger.info(f"复制测试图片: {test_image_source} -> {input_file}")
    
    # 创建任务数据
    task_data = {
        "input_files": [str(input_file)],
        "output_dir": str(output_dir),
        "style": "标准",  # 基本风格，应该在所有安装中都可用
        "count": 1,  # 生成1张以加快测试
        "multiplier_style": 0.25,
        "use_pose": False,
        "pose_file": None,
    }
    
    # 添加任务到队列
    logger.info("添加任务到队列...")
    add_task(task_id, task_data)
    
    # 检查任务状态
    task = get_task_status(task_id)
    logger.info(f"初始任务状态: {task.get('status')}")
    
    # 等待任务处理完成
    logger.info("等待任务处理...")
    timeout = 300  # 5分钟超时
    start_time = time.time()
    
    while True:
        task = get_task_status(task_id)
        if not task:
            logger.error("任务不存在!")
            return False
        
        status = task.get("status")
        progress = task.get("progress", 0)
        
        logger.info(f"任务状态: {status}, 进度: {progress}%")
        
        if status == TaskStatus.COMPLETED:
            logger.info("任务完成!")
            result_files = task.get("result", [])
            logger.info(f"生成了 {len(result_files)} 张图片")
            for file in result_files:
                logger.info(f"  - {file}")
            return True
        
        if status == TaskStatus.FAILED:
            logger.error(f"任务失败: {task.get('message', '未知错误')}")
            return False
        
        # 检查超时
        if time.time() - start_time > timeout:
            logger.error(f"任务处理超时 ({timeout}秒)")
            return False
        
        # 等待一段时间再检查
        await asyncio.sleep(5)

async def main():
    """主函数"""
    os.makedirs(project_root / "logs", exist_ok=True)
    logger.info("=== 开始运行FaceChain队列测试 ===")
    
    try:
        success = await test_facechain_queue()
        if success:
            logger.info("✓ 测试成功完成!")
        else:
            logger.error("✗ 测试失败!")
    except Exception as e:
        logger.exception(f"测试过程中发生错误: {e}")
    
    logger.info("=== FaceChain队列测试结束 ===")

if __name__ == "__main__":
    asyncio.run(main()) 