# FaceChain Celery Worker启动脚本

<#
.SYNOPSIS
    启动Monna AI Celery Worker服务
.DESCRIPTION
    该脚本激活Python 3.10虚拟环境并启动Celery Worker，
    用于处理Redis队列中的FaceChain任务。
.NOTES
    版本:      1.0
    作者:      Monna AI团队
    日期:      2023-10-25
#>

# 检查虚拟环境是否存在
$venvPath = ".\venv310"
if (-not (Test-Path "$venvPath\Scripts\activate.ps1")) {
    Write-Host "错误: 未找到Python 3.10虚拟环境，请确保在正确的目录中运行脚本。" -ForegroundColor Red
    exit 1
}

# 设置环境变量
$env:PYTHONIOENCODING = "utf-8"
$env:C_FORCE_ROOT = "true"  # 允许以root用户运行Celery (对于Windows可能不需要)

# 显示欢迎信息
Write-Host "正在启动Monna AI Celery Worker服务..." -ForegroundColor Green
Write-Host "该服务将处理Redis队列中的FaceChain任务" -ForegroundColor Cyan
Write-Host "------------------------------------------------" -ForegroundColor Gray

# 激活虚拟环境并启动服务
try {
    # 激活虚拟环境
    Write-Host "激活Python 3.10虚拟环境..." -ForegroundColor Cyan
    & "$venvPath\Scripts\activate"
    
    # 启动Celery Worker
    Write-Host "启动Celery Worker..." -ForegroundColor Cyan
    python --version
    
    # 使用 celery 命令启动worker
    # -A 指定Celery应用
    # --loglevel 设置日志级别
    # -Q 指定要处理的队列
    # --concurrency 设置并发处理的任务数
    
    celery -A app.worker.celery_app worker --loglevel=info -Q facechain_tasks --concurrency=1
    
    # 如果服务退出，捕获错误码
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: Celery Worker异常退出，错误码 $LASTEXITCODE" -ForegroundColor Red
    }
}
catch {
    Write-Host "发生错误: $_" -ForegroundColor Red
}
finally {
    # 脚本结束时确保虚拟环境被正确停用
    if (Get-Command deactivate -ErrorAction SilentlyContinue) {
        deactivate
    }
} 