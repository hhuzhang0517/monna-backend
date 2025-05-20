# 启动AI照片生成Worker的PowerShell脚本

Write-Host "Starting AI photo generation worker..." -ForegroundColor Cyan

# 检查Python版本
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Python not found. Please install Python 3.8+ and try again." -ForegroundColor Red
    exit 1
}

# 设置和检查虚拟环境
$venvPath = "../../.venv"  # 相对于脚本位置的路径
if (Test-Path -Path $venvPath) {
    Write-Host "Found virtual environment, activating..." -ForegroundColor Green
    try {
        # 活跃虚拟环境
        & "$venvPath/Scripts/Activate.ps1"
    }
    catch {
        Write-Host "Failed to activate virtual environment. Continuing with system Python..." -ForegroundColor Yellow
    }
}

# 检查并安装Python依赖
Write-Host "Checking and installing Python dependencies..." -ForegroundColor Cyan
pip install redis torch torchvision pillow numpy tqdm matplotlib requests scikit-image

# 设置默认环境变量（不覆盖已有设置）
if (-not $env:REDIS_HOST) { $env:REDIS_HOST = "localhost" }
if (-not $env:REDIS_PORT) { $env:REDIS_PORT = "6379" }
if (-not $env:REDIS_DB) { $env:REDIS_DB = "0" }

Write-Host "Starting Python Worker..." -ForegroundColor Green
python worker.py  # 启动Python Worker
