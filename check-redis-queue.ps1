# Redis队列检查脚本

<#
.SYNOPSIS
    检查Redis队列状态
.DESCRIPTION
    该脚本激活Python 3.10虚拟环境并运行Redis队列检查脚本，
    用于诊断Redis队列问题。
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

# 显示欢迎信息
Write-Host "正在检查Redis队列状态..." -ForegroundColor Green
Write-Host "------------------------------------------------" -ForegroundColor Gray

# 激活虚拟环境并运行脚本
try {
    # 激活虚拟环境
    Write-Host "激活Python 3.10虚拟环境..." -ForegroundColor Cyan
    & "$venvPath\Scripts\activate"
    
    # 运行Redis检查脚本
    Write-Host "运行Redis队列检查脚本..." -ForegroundColor Cyan
    python check-redis-queue.py
    
    # 如果脚本退出，捕获错误码
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: Redis队列检查脚本异常退出，错误码 $LASTEXITCODE" -ForegroundColor Red
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
    
    Write-Host "------------------------------------------------" -ForegroundColor Gray
    Write-Host "检查完成。按任意键退出..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
} 