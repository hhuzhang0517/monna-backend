# Android模拟器连接检测脚本
Write-Host "检查Android模拟器网络连接配置..." -ForegroundColor Green

# 检查API服务是否运行
Write-Host "检查API服务是否运行:" -ForegroundColor Cyan
$apiRunning = netstat -an | Select-String -Pattern ":3000"
if ($apiRunning) {
    Write-Host "API服务正在运行（端口3000）" -ForegroundColor Green
    $apiRunning
} else {
    Write-Host "API服务未运行，请先启动API服务" -ForegroundColor Red
}

# 检查证书文件
$CERT_PATH = "D:/xroting/monna/certificate.crt"
if (Test-Path $CERT_PATH) {
    Write-Host "证书文件存在: $CERT_PATH" -ForegroundColor Green
} else {
    Write-Host "证书文件不存在！" -ForegroundColor Red
}

# 检查Android模拟器访问的特殊IP
Write-Host "`nAndroid模拟器特殊IP配置:" -ForegroundColor Cyan
Write-Host "Android模拟器应使用 10.0.2.2 而不是 192.168.10.105 访问主机" -ForegroundColor Yellow
Write-Host "请确保在API配置中针对Android平台使用了正确的IP" -ForegroundColor Yellow

# 检查远程访问配置
Write-Host "`n防火墙检查:" -ForegroundColor Cyan
try {
    $rules = Get-NetFirewallRule -DisplayName "*3000*" -ErrorAction SilentlyContinue
    if ($rules) {
        Write-Host "找到端口3000的防火墙规则:" -ForegroundColor Green
        $rules | Format-Table Name,DisplayName,Enabled,Direction,Action -AutoSize
    } else {
        Write-Host "未找到针对端口3000的防火墙规则，这可能会阻止外部访问" -ForegroundColor Yellow
        Write-Host "可以使用以下命令添加规则:" -ForegroundColor Yellow
        Write-Host "New-NetFirewallRule -DisplayName 'Monna API Port 3000' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3000" -ForegroundColor Gray
    }
} catch {
    Write-Host "无法检查防火墙规则: $_" -ForegroundColor Red
}

# 提供解决方案建议
Write-Host "`n解决方案建议:" -ForegroundColor Cyan
Write-Host "1. 在Android模拟器中使用 https://10.0.2.2:3000/api/docs 而不是 192.168.10.105" -ForegroundColor Green
Write-Host "2. 确保network_security_config.xml中配置了信任10.0.2.2的证书" -ForegroundColor Green
Write-Host "3. 在monna-app/src/services/api.js中针对Android平台特殊处理IP地址" -ForegroundColor Green
Write-Host "4. 重新打包并安装更新后的应用到模拟器" -ForegroundColor Green
Write-Host "5. 如果仍然无法连接，请考虑在模拟器中使用adb命令或设置代理" -ForegroundColor Green 