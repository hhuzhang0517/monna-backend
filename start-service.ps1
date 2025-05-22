# 简化版FaceChain后端服务启动脚本

# 设置环境变量
$env:PYTHONIOENCODING = "utf-8"

# 显示启动信息
Write-Host "正在启动Monna AI FaceChain后端服务..." -ForegroundColor Green

# 虚拟环境路径
$venvPath = ".\venv310\Scripts\activate.ps1"

# 如果虚拟环境存在，则激活它
if (Test-Path $venvPath) {
    Write-Host "激活Python虚拟环境..." -ForegroundColor Cyan
    & $venvPath
    
    # 启动FastAPI服务
    Write-Host "启动FastAPI服务..." -ForegroundColor Cyan
    python main.py
} else {
    Write-Host "错误: 未找到Python虚拟环境: $venvPath" -ForegroundColor Red
    Write-Host "请确保已安装并配置Python 3.10虚拟环境" -ForegroundColor Yellow
} 