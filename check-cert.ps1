# 证书验证脚本
Write-Host "验证SSL证书配置..." -ForegroundColor Green

$CERT_PATH = "D:/xroting/monna/certificate.crt"
$KEY_PATH = "D:/xroting/monna/private.key"

# 检查证书文件是否存在
if (-not (Test-Path $CERT_PATH)) {
    Write-Host "证书文件不存在: $CERT_PATH" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $KEY_PATH)) {
    Write-Host "私钥文件不存在: $KEY_PATH" -ForegroundColor Red
    exit 1
}

Write-Host "证书文件已找到" -ForegroundColor Green

# 使用OpenSSL验证证书
try {
    Write-Host "证书信息:" -ForegroundColor Cyan
    openssl x509 -in $CERT_PATH -text -noout | Select-String -Pattern "Subject:|Issuer:|Not Before:|Not After"
} catch {
    Write-Host "无法验证证书内容: $_" -ForegroundColor Red
}

# 检查本机IP地址
Write-Host "`n本机IP地址:" -ForegroundColor Cyan
ipconfig | Select-String -Pattern "IPv4"

# 端口检查
Write-Host "`n检查服务器端口状态:" -ForegroundColor Cyan
netstat -an | Select-String -Pattern ":3000"

Write-Host "`n如果需在移动设备上访问，请确保:" -ForegroundColor Yellow
Write-Host "1. 移动设备与服务器处于同一网络" -ForegroundColor Yellow
Write-Host "2. 服务器防火墙已开放3000端口" -ForegroundColor Yellow
Write-Host "3. 前端配置中的apiHost设置为正确的服务器IP地址" -ForegroundColor Yellow
Write-Host "4. 使用开发环境切换按钮测试连接" -ForegroundColor Yellow 