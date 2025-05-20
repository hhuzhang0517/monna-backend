# 启动FaceChain前端兼容代理服务器
Write-Host "正在启动前端兼容代理服务器..." -ForegroundColor Green

# 激活Python虚拟环境
if (Test-Path "./venv310/Scripts/Activate.ps1") {
    & ./venv310/Scripts/Activate.ps1
    Write-Host "已激活虚拟环境 venv310" -ForegroundColor Green
} else {
    Write-Host "警告: 虚拟环境不存在，尝试使用系统Python" -ForegroundColor Yellow
}

# 确保必要的依赖已安装
pip install fastapi uvicorn requests

# 创建必要的目录
$logDir = "./logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
    Write-Host "创建日志目录: $logDir" -ForegroundColor Green
}

# 启动代理服务器
Write-Host "启动代理服务器在端口 8080..." -ForegroundColor Green
Write-Host "前端应访问 http://localhost:8080/api/v1/facechain/tasks/test-id 来测试连接" -ForegroundColor Cyan

python proxy-server.py 