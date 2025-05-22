# FaceChain后端服务启动脚本

<#
.SYNOPSIS
    启动Monna AI FaceChain后端服务
.DESCRIPTION
    该脚本激活Python 3.10虚拟环境并启动FastAPI后端服务，
    包括任务队列系统和FaceChain AI模型初始化。
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

# 激活虚拟环境并启动服务
Write-Host "正在启动Monna AI FaceChain后端服务..." -ForegroundColor Green
Write-Host "激活Python 3.10虚拟环境..." -ForegroundColor Cyan

try {
    & "$venvPath\Scripts\activate"
    
    # 检查activation是否成功
    #if ($LASTEXITCODE -ne 0) {
     #   Write-Host "错误: 虚拟环境激活失败。" -ForegroundColor Red
    #    exit 1
    #}
    
    Write-Host "启动FastAPI服务..." -ForegroundColor Cyan
    python --version
    python main.py
    
    # 如果服务退出，捕获错误码
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 服务异常退出，错误码 $LASTEXITCODE" -ForegroundColor Red
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