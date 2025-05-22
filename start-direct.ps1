# FaceChain后端服务启动脚本 (直接版本)

<#
.SYNOPSIS
    启动Monna AI FaceChain后端服务
.DESCRIPTION
    该脚本使用Python解释器直接启动FastAPI后端服务，
    不依赖虚拟环境激活命令，更适合可能遇到PowerShell执行策略限制的环境。
.NOTES
    版本:      1.1
    作者:      Monna AI团队
    日期:      2025-05-19
#>

# 设置变量
$pythonExecutable = "python"  # 使用系统PATH中的Python
$scriptPath = Join-Path $PSScriptRoot "main.py"  # 主脚本路径

# 输出标题
Write-Host "正在启动Monna AI FaceChain后端服务..." -ForegroundColor Green

# 检查是否能找到Python解释器
try {
    $pythonVersion = & $pythonExecutable --version
    Write-Host "使用Python解释器: $pythonVersion" -ForegroundColor Cyan
} catch {
    Write-Host "错误: 无法找到Python解释器。请确保Python已安装并添加到PATH环境变量中。" -ForegroundColor Red
    exit 1
}

# 设置环境变量
$env:PYTHONIOENCODING = "utf-8"

# 启动服务
try {
    Write-Host "启动FastAPI服务..." -ForegroundColor Cyan
    Write-Host "执行命令: $pythonExecutable $scriptPath" -ForegroundColor Gray
    
    # 使用Python解释器直接运行main.py
    & $pythonExecutable $scriptPath
    
    # 捕获退出代码
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 服务异常退出，错误码 $LASTEXITCODE" -ForegroundColor Red
    }
} catch {
    Write-Host "发生错误: $_" -ForegroundColor Red
} 