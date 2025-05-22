# -*- coding: utf-8 -*-
# 启动所有Monna AI服务的脚本

<#
.SYNOPSIS
    启动Monna AI所有后端服务
.DESCRIPTION
    该脚本启动Node.js API服务、Python FaceChain服务和Celery Worker服务，
    使用单独的窗口运行每个服务，便于查看日志和独立管理。
.NOTES
    版本:      1.2
    作者:      Monna AI团队
    日期:      2025-05-21
#>

# 设置变量和颜色
$scriptRoot = $PSScriptRoot
$nodeServiceScript = Join-Path $scriptRoot "run-dev.ps1"
# 使用start-facechain-service.ps1替代start-direct.ps1，因为前者会激活虚拟环境
$pythonServiceScript = Join-Path $scriptRoot "start-facechain-service.ps1"
$celeryWorkerScript = Join-Path $scriptRoot "start-celery-worker.ps1"
$pythonVenvPath = Join-Path $scriptRoot "venv310"

function Write-ColoredText {
    param (
        [string]$message,
        [string]$color = "White"
    )
    Write-Host $message -ForegroundColor $color
}

# 显示欢迎信息
Write-ColoredText "正在启动Monna AI全部后端服务..." "Green"
Write-ColoredText "这将在单独的窗口中启动Node.js API服务、Python FaceChain服务和Celery Worker服务" "Cyan"
Write-ColoredText "------------------------------------------------" "Gray"

# 检查虚拟环境是否存在
if (-not (Test-Path $pythonVenvPath)) {
    Write-ColoredText "错误: 未找到Python虚拟环境: $pythonVenvPath" "Red"
    Write-ColoredText "请确保已安装并配置Python 3.10虚拟环境" "Yellow"
    exit 1
}

# 检查激活脚本是否存在
$activateScript = Join-Path $pythonVenvPath "Scripts\activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-ColoredText "错误: 未找到虚拟环境激活脚本: $activateScript" "Red"
    exit 1
}

# 检查脚本是否存在
if (-not (Test-Path $nodeServiceScript)) {
    Write-ColoredText "错误: 未找到Node.js API服务启动脚本: $nodeServiceScript" "Red"
    exit 1
}

if (-not (Test-Path $pythonServiceScript)) {
    Write-ColoredText "错误: 未找到Python FaceChain服务启动脚本: $pythonServiceScript" "Red"
    exit 1
}

if (-not (Test-Path $celeryWorkerScript)) {
    Write-ColoredText "错误: 未找到Celery Worker服务启动脚本: $celeryWorkerScript" "Red"
    exit 1
}

# 启动Node.js API服务
Write-ColoredText "正在启动Node.js API服务..." "Yellow"
try {
    Start-Process powershell.exe -ArgumentList "-NoExit", "-File", "`"$nodeServiceScript`"" -WindowStyle Normal
    Write-ColoredText "Node.js API服务启动成功" "Green"
} catch {
    Write-ColoredText "启动Node.js API服务时出错: $_" "Red"
}

# 等待几秒钟让Node.js API服务启动
Write-ColoredText "等待Node.js API服务启动..." "Gray"
Start-Sleep -Seconds 5

# 启动Python FaceChain服务
Write-ColoredText "正在启动Python FaceChain服务..." "Yellow"
try {
    # 创建使用虚拟环境的PowerShell命令
    $pythonCommand = @"
Set-Location "$scriptRoot"
& "$activateScript"
& "$pythonServiceScript"
"@
    # 将命令保存为临时文件
    $tempScriptPath = Join-Path $env:TEMP "start-python-with-venv.ps1"
    $pythonCommand | Out-File -FilePath $tempScriptPath -Encoding utf8
    
    # 启动PowerShell窗口执行临时脚本
    Start-Process powershell.exe -ArgumentList "-NoExit", "-File", "`"$tempScriptPath`"" -WindowStyle Normal
    Write-ColoredText "Python FaceChain服务启动成功" "Green"
} catch {
    Write-ColoredText "启动Python FaceChain服务时出错: $_" "Red"
}

# 等待几秒钟让Python FaceChain服务启动
Write-ColoredText "等待Python FaceChain服务启动..." "Gray"
Start-Sleep -Seconds 5

# 启动Celery Worker服务
Write-ColoredText "正在启动Celery Worker服务..." "Yellow"
try {
    # 创建使用虚拟环境的PowerShell命令
    $celeryCommand = @"
Set-Location "$scriptRoot"
& "$activateScript"
& "$celeryWorkerScript"
"@
    # 将命令保存为临时文件
    $tempCeleryScriptPath = Join-Path $env:TEMP "start-celery-with-venv.ps1"
    $celeryCommand | Out-File -FilePath $tempCeleryScriptPath -Encoding utf8
    
    # 启动PowerShell窗口执行临时脚本
    Start-Process powershell.exe -ArgumentList "-NoExit", "-File", "`"$tempCeleryScriptPath`"" -WindowStyle Normal
    Write-ColoredText "Celery Worker服务启动成功" "Green"
} catch {
    Write-ColoredText "启动Celery Worker服务时出错: $_" "Red"
}

Write-ColoredText "------------------------------------------------" "Gray"
Write-ColoredText "服务已在独立窗口中启动，请勿关闭此窗口" "Yellow"
Write-ColoredText "Node.js API服务运行在: http://localhost:3000/api/docs" "Cyan"
Write-ColoredText "Python FaceChain服务运行在: http://localhost:8000/api/v1/docs" "Cyan"
Write-ColoredText "Celery Worker服务已启动并监听Redis队列中的任务" "Cyan"
Write-ColoredText "------------------------------------------------" "Gray"

# 保持此窗口打开，显示服务状态
while ($true) {
    # 检查Node.js API服务
    try {
        $nodeApiStatus = Invoke-WebRequest -Uri "http://localhost:3000/api" -TimeoutSec 2 -ErrorAction SilentlyContinue
        $nodeApiRunning = $nodeApiStatus.StatusCode -eq 200
    } catch {
        $nodeApiRunning = $false
    }
    
    # 检查Python FaceChain服务
    try {
        $pythonApiStatus = Invoke-WebRequest -Uri "http://localhost:8000/api/v1/facechain/connection-test" -TimeoutSec 2 -ErrorAction SilentlyContinue
        $pythonApiRunning = $pythonApiStatus.StatusCode -eq 200
    } catch {
        $pythonApiRunning = $false
    }
    
    # 显示状态
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $nodeStatus = if ($nodeApiRunning) { "运行中" } else { "已停止" }
    $pythonStatus = if ($pythonApiRunning) { "运行中" } else { "已停止" }
    
    Write-Host "`r[$timestamp] Node.js API: " -NoNewline
    Write-Host $nodeStatus -NoNewline -ForegroundColor $(if ($nodeApiRunning) { "Green" } else { "Red" })
    Write-Host " | Python FaceChain: " -NoNewline
    Write-Host $pythonStatus -NoNewline -ForegroundColor $(if ($pythonApiRunning) { "Green" } else { "Red" })
    Write-Host " | Celery Worker: 运行中" -NoNewline -ForegroundColor "Green"
    Write-Host " (按Ctrl+C退出)" -NoNewline
    
    # 等待10秒再检查
    Start-Sleep -Seconds 10
}