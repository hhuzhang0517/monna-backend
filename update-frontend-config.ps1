# 更新前端环境配置脚本
Write-Host "正在更新前端环境配置..." -ForegroundColor Green

$frontendPath = "../monna-app"

# 检查前端目录是否存在
if (-not (Test-Path $frontendPath)) {
    Write-Host "错误: 前端目录 $frontendPath 不存在" -ForegroundColor Red
    exit 1
}

# 查找可能的环境配置文件
$configFiles = @(
    "$frontendPath/.env",
    "$frontendPath/.env.local",
    "$frontendPath/.env.development",
    "$frontendPath/src/config/env.js",
    "$frontendPath/src/config/environment.js",
    "$frontendPath/src/services/api.js",
    "$frontendPath/src/constants/api.js"
)

$foundConfig = $false

foreach ($file in $configFiles) {
    if (Test-Path $file) {
        Write-Host "找到配置文件: $file" -ForegroundColor Green
        
        # 读取文件内容
        $content = Get-Content $file -Raw
        
        # 备份原始文件
        Copy-Item $file "$file.bak"
        Write-Host "已创建配置文件备份: $file.bak" -ForegroundColor Yellow
        
        # 替换端口配置
        if ($content -match "PYTHON_FACECHAIN_DEV_PORT\s*=\s*8000") {
            $content = $content -replace "PYTHON_FACECHAIN_DEV_PORT\s*=\s*8000", "PYTHON_FACECHAIN_DEV_PORT = 8080"
            Set-Content $file $content
            Write-Host "已将端口配置从8000更新为8080" -ForegroundColor Green
            $foundConfig = $true
        }
        elseif ($content -match "PYTHON_FACECHAIN_DEV_PORT\s*:\s*['\"]?8000['\"]?") {
            $content = $content -replace "PYTHON_FACECHAIN_DEV_PORT\s*:\s*['\"]?8000['\"]?", "PYTHON_FACECHAIN_DEV_PORT: '8080'"
            Set-Content $file $content
            Write-Host "已将端口配置从8000更新为8080" -ForegroundColor Green
            $foundConfig = $true
        }
        elseif ($content -match "pythonEndpoints") {
            # 如果文件是api.js并包含测试端点信息
            $content = $content -replace "http://\$\{PYTHON_FACECHAIN_DEV_HOST\}:8000/", "http://`${PYTHON_FACECHAIN_DEV_HOST}:8080/"
            Set-Content $file $content
            Write-Host "已将pythonEndpoints中的端口配置从8000更新为8080" -ForegroundColor Green
            $foundConfig = $true
        }
    }
}

if ($foundConfig) {
    Write-Host "前端环境配置已成功更新！" -ForegroundColor Green
    Write-Host "现在FaceChain API请求将发送到端口8080的代理服务器" -ForegroundColor Cyan
} else {
    Write-Host "警告: 未找到可更新的环境配置文件" -ForegroundColor Yellow
    Write-Host "请手动将前端配置中的FaceChain API端口从8000更改为8080" -ForegroundColor Yellow
}

Write-Host "配置更新完成。请重启前端应用以应用新配置。" -ForegroundColor Green 