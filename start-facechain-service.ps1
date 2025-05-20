# FaceChain服务启动脚本 (ASCII编码)
Write-Host "正在启动FaceChain服务..." -ForegroundColor Green

# 激活Python虚拟环境
$venvPath = "./venv310/Scripts/Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "已激活虚拟环境 venv310" -ForegroundColor Green
} else {
    Write-Host "警告: 虚拟环境不存在，尝试使用系统Python" -ForegroundColor Yellow
}

# 创建必要的目录
$logDir = "./logs"
$uploadDir = "./data/uploads/facechain"
$outputDir = "./data/outputs"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
    Write-Host "创建日志目录: $logDir" -ForegroundColor Green
}

if (-not (Test-Path $uploadDir)) {
    New-Item -ItemType Directory -Path $uploadDir -Force
    Write-Host "创建上传目录: $uploadDir" -ForegroundColor Green
}

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
    Write-Host "创建输出目录: $outputDir" -ForegroundColor Green
}

# 检查Python模块是否可用
Write-Host "检查Python模块依赖..." -ForegroundColor Green
python -c "import datasets; print(f'datasets version: {datasets.__version__}')" 
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 缺少必要的Python模块 'datasets'，请运行: pip install datasets==2.16.0" -ForegroundColor Red
    exit 1
}

# 启动FastAPI应用程序 - 使用Python直接执行而不是uvicorn命令
Write-Host "启动FastAPI应用..." -ForegroundColor Green
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload 