# FaceChain任务队列测试脚本

<#
.SYNOPSIS
    测试Monna AI FaceChain任务队列功能
.DESCRIPTION
    该脚本激活Python 3.10虚拟环境并运行任务队列测试，
    测试FaceChain模型和任务队列系统的完整流程。
.NOTES
    版本:      1.0
    作者:      Monna AI团队
    日期:      2023-10-01
#>

# 检查虚拟环境是否存在
$venvPath = ".\venv310"
if (-not (Test-Path "$venvPath\Scripts\activate.ps1")) {
    Write-Host "错误: 未找到Python 3.10虚拟环境，请确保在正确的目录中运行脚本。" -ForegroundColor Red
    exit 1
}

# 设置环境变量
$env:PYTHONIOENCODING = "utf-8"

# 激活虚拟环境并运行测试
Write-Host "正在启动Monna AI FaceChain任务队列测试..." -ForegroundColor Green
Write-Host "激活Python 3.10虚拟环境..." -ForegroundColor Cyan

try {
    & "$venvPath\Scripts\activate.ps1"
    
    # 检查activation是否成功
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 虚拟环境激活失败。" -ForegroundColor Red
        exit 1
    }
    
    # 确保test-images目录存在并有测试图片
    if (-not (Test-Path ".\test-images")) {
        New-Item -ItemType Directory -Path ".\test-images" -Force
        Write-Host "创建测试图片目录: .\test-images" -ForegroundColor Yellow
        Write-Host "注意: 请确保test-images目录中至少有一张JPG或PNG图片" -ForegroundColor Yellow
    }
    
    Write-Host "运行FaceChain任务队列测试..." -ForegroundColor Cyan
    python test_facechain_queue.py
    
    # 如果测试退出，捕获错误码
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 测试失败，错误码 $LASTEXITCODE" -ForegroundColor Red
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