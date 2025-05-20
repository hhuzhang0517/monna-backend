import redis
import json
import os
import uuid
from pathlib import Path
import glob
import time
import shutil

# Redis连接参数
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_QUEUE = 'ai_tasks'

print('模拟创建AI照片任务...')

# 强制创建一个新的随机任务ID
task_id = str(uuid.uuid4())
print(f'新任务ID: {task_id}')

# 连接Redis
client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# 使用现有照片目录的照片
base_dir = Path('..').resolve()
upload_dirs = list(base_dir.glob('data/uploads/*'))

if upload_dirs:
    # 选择第一个上传目录
    sample_dir = upload_dirs[0]
    old_task_id = sample_dir.name
    print(f'使用照片来源目录: {sample_dir}')
    
    # 创建新的上传目录
    upload_dir = base_dir / 'data' / 'uploads' / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制照片到新目录
    photos = []
    
    # 找到所有照片
    source_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        source_files.extend(list(sample_dir.glob(ext)))
    
    if not source_files:
        print('错误: 上传目录中没有找到照片')
        exit(1)
    
    # 复制照片到新目录
    for i, src_file in enumerate(source_files):
        dest = upload_dir / f'sample_{i+1}{src_file.suffix}'
        shutil.copy(src_file, dest)
        photos.append(str(dest))
    
    print(f'创建新上传目录: {upload_dir}')
    print(f'复制 {len(photos)} 张照片用于测试')
    
    # 清理旧的输出目录（如果存在）
    output_dir = base_dir / 'data' / 'outputs' / task_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f'清理旧输出目录: {output_dir}')
else:
    print('错误: 没有找到现有上传目录')
    exit(1)

# 准备任务数据
# 尝试使用不同风格: 婚纱, 油画, 古风, 漫画
styles = ['古风', '油画', '婚纱', '漫画']  # 测试所有风格

task_data = {
    'taskId': task_id,
    'taskType': 'photo',
    'styles': styles,
    'filePaths': photos,
    'createdAt': time.strftime('%Y-%m-%dT%H:%M:%S')
}

# 转为JSON
task_json = json.dumps(task_data)

# 发送到Redis队列
client.rpush(REDIS_QUEUE, task_json)

print(f'成功将任务 {task_id} 推送到Redis队列，风格: {styles}')
print('现在Python Worker应该会自动处理这个任务')
print(f'结果将保存到: {base_dir}/data/outputs/{task_id}/') 