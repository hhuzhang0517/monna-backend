# 开发环境启动脚本
Write-Host "正在启动开发环境服务器..." -ForegroundColor Green

# 设置环境变量
$env:NODE_ENV = "development"
$env:USE_HTTPS = "true" # 启用HTTPS
$env:SSL_CERT_PATH = "D:/xroting/monna/certificate.crt"
$env:SSL_KEY_PATH = "D:/xroting/monna/private.key"

# 切换到API目录
cd ./node-api

# 使用npm启动服务
npm run start:dev:win

# 如果想启用HTTPS，请使用以下命令生成证书，然后将USE_HTTPS设为true
# node generate-certs.js 