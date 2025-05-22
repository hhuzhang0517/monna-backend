# 测试脚本 - 手动创建任务到Redis队列
# 这个脚本用于测试Python Worker的处理功能
# 使用方法: .\test-task.ps1

Write-Host "模拟创建AI照片任务..." -ForegroundColor Cyan

# 加载必要的Python库
$pythonCode = @"
import redis
import json
import os
import uuid
from pathlib import Path
import glob

# Redis连接参数
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_QUEUE = 'ai_tasks'

# 创建一个随机的任务ID
task_id = str(uuid.uuid4())
print(f'创建任务ID: {task_id}')

# 连接Redis
client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

# 检查上传目录是否存在照片可用于测试
base_dir = Path('..').resolve()
upload_dirs = list(base_dir.glob('data/uploads/*'))

if upload_dirs:
    # 使用现有上传目录中的照片
    sample_dir = upload_dirs[0]
    task_id = sample_dir.name
    print(f'使用现有上传目录: {sample_dir}')
    photos = []
    
    # 收集目录中的所有图片
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        photos.extend([str(p) for p in sample_dir.glob(ext)])
    
    if not photos:
        print('错误: 上传目录中没有找到照片')
        exit(1)
else:
    # 创建一个测试目录并复制一些示例照片
    print('没有找到现有上传目录，创建新测试任务')
    
    # 检查是否有示例照片
    example_photos = list(Path().glob('test-photos/*.jpg'))
    if not example_photos:
        print('错误: 未找到测试照片，请在test-photos目录中放置一些jpg照片')
        exit(1)
    
    # 创建上传目录
    upload_dir = base_dir / 'data' / 'uploads' / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # 复制示例照片到上传目录
    photos = []
    for i, photo in enumerate(example_photos[:8]):  # 最多使用8张照片
        dest = upload_dir / f'sample_{i+1}.jpg'
        import shutil
        shutil.copy(photo, dest)
        photos.append(str(dest))
    
    print(f'创建测试上传目录: {upload_dir}')
    print(f'复制 {len(photos)} 张照片用于测试')

# 准备任务数据
task_data = {
    'taskId': task_id,
    'taskType': 'photo',
    'styles': ['婚纱'],  # 可以使用其他风格: 油画, 古风, 漫画
    'filePaths': photos,
    'createdAt': None  # Python会自动填充当前时间
}

# 转为JSON
task_json = json.dumps(task_data)

# 发送到Redis队列
client.rpush(REDIS_QUEUE, task_json)

print(f'成功将任务 {task_id} 推送到Redis队列')
print('现在Python Worker应该会自动处理这个任务')
print(f'结果将保存到: {base_dir}/data/outputs/{task_id}/')
"@

# 创建临时Python脚本
$tempFile = New-TemporaryFile
$tempPyFile = $tempFile.FullName + ".py"
Move-Item -Path $tempFile -Destination $tempPyFile
Set-Content -Path $tempPyFile -Value $pythonCode

# 执行Python脚本
Write-Host "执行Python测试脚本..." -ForegroundColor Yellow
python $tempPyFile

# 清理临时文件
Remove-Item -Path $tempPyFile

Write-Host "`n等待Python Worker处理任务..." -ForegroundColor Green
Write-Host "请查看Python Worker窗口的日志输出" -ForegroundColor Green 