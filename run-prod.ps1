# 生产环境启动脚本
Write-Host "正在启动生产环境服务器..." -ForegroundColor Green

# 设置环境变量
$env:NODE_ENV = "production"
$env:USE_HTTPS = "true" # 生产环境使用HTTPS

# 切换到API目录
cd ./node-api

# 使用npm启动服务
npm run start:prod:win 