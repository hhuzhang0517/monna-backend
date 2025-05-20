# SSL证书配置指南 - Monna AI项目

本文档提供了如何在Monna AI项目中配置和切换SSL证书的详细说明，适用于开发测试环境和生产环境。

## 目录

1. [SSL证书概述](#ssl证书概述)
2. [开发测试环境配置](#开发测试环境配置)
3. [生产环境配置](#生产环境配置)
4. [环境切换方法](#环境切换方法)
5. [证书路径配置](#证书路径配置)
6. [手动生成自签名证书](#手动生成自签名证书)
7. [问题排查](#问题排查)

## SSL证书概述

Monna AI项目使用HTTPS协议保证数据传输安全。我们的配置支持两种环境：

- **开发测试环境**: 使用自签名SSL证书，适用于本地开发和测试
- **生产环境**: 使用由可信CA机构颁发的SSL证书，适用于线上部署

## 开发测试环境配置

### 配置步骤

1. 将自签名证书文件保存在指定位置:
   - 证书文件(CRT): `D:/xroting/monna/certificate.crt`
   - 私钥文件(KEY): `D:/xroting/monna/private.key`

2. 确保环境变量配置正确:
   ```
   NODE_ENV=development
   USE_HTTPS=true
   SSL_CERT_PATH=D:/xroting/monna/certificate.crt
   SSL_KEY_PATH=D:/xroting/monna/private.key
   ```

3. 使用开发环境启动命令:
   ```bash
   # Windows
   npm run start:dev:win
   
   # Linux/Mac
   npm run start:dev
   ```

## 生产环境配置

### 配置步骤

1. 获取由可信CA机构颁发的SSL证书，并将其保存在安全位置。
   示例路径:
   - 证书文件(CRT): `/etc/ssl/monna/certificate.crt`
   - 私钥文件(KEY): `/etc/ssl/monna/private.key`

2. 配置环境变量:
   ```
   NODE_ENV=production
   USE_HTTPS=true
   SSL_CERT_PATH=/etc/ssl/monna/certificate.crt
   SSL_KEY_PATH=/etc/ssl/monna/private.key
   ```

3. 使用生产环境启动命令:
   ```bash
   # Windows
   npm run start:prod:win
   
   # Linux/Mac
   npm run start:prod
   ```

## 环境切换方法

### 后端环境切换

切换环境只需运行对应的启动命令:

```bash
# 开发环境 (Windows)
npm run start:dev:win

# 生产环境 (Windows)
npm run start:prod:win

# 开发环境 (Linux/Mac)
npm run start:dev

# 生产环境 (Linux/Mac)
npm run start:prod
```

### 前端环境切换

1. **开发模式中**：可使用界面右下角的"ENV"按钮切换环境
2. **代码配置**：可修改`monna-app/src/services/api.js`中的环境配置

## 证书路径配置

SSL证书路径可通过两种方式配置:

1. **环境变量**: 通过`.env`文件或系统环境变量设置:
   ```
   SSL_CERT_PATH=path/to/certificate.crt
   SSL_KEY_PATH=path/to/private.key
   ```

2. **配置文件**: 编辑`.env.development`或`.env.production`文件中的证书路径

## 手动生成自签名证书

如需生成新的自签名证书，可使用以下命令:

### Windows (使用OpenSSL)

```bash
# 生成私钥
openssl genrsa -out private.key 2048

# 生成自签名证书
openssl req -new -x509 -key private.key -out certificate.crt -days 365 -subj "/CN=localhost"
```

### Linux/Mac

```bash
# 生成私钥
openssl genrsa -out private.key 2048

# 生成证书签名请求(CSR)
openssl req -new -key private.key -out csr.pem

# 生成自签名证书
openssl x509 -req -days 365 -in csr.pem -signkey private.key -out certificate.crt
```

## 问题排查

### 常见问题

1. **证书找不到错误**:
   ```
   Error: ENOENT: no such file or directory, open 'path/to/certificate.crt'
   ```
   
   解决方法: 检查证书路径是否正确，确保文件存在且有读取权限。

2. **移动设备无法连接**:
   
   解决方法: 确保移动设备与服务器在同一网络，且使用正确的服务器IP地址。
   
   前端配置修改: `monna-app/src/services/api.js` 中修改 `apiHost` 为正确的IP地址。

3. **自签名证书警告**:
   
   这是正常的，因为浏览器无法验证自签名证书。开发环境中可以忽略此警告。

### 移动设备SSL配置

Android设备可能需要额外配置:

1. 确保 `monna-app/android/app/src/main/res/xml/network_security_config.xml` 中包含服务器域名
2. 确保 `monna-app/android/app/src/main/AndroidManifest.xml` 中引用了安全配置

iOS设备设置:

1. `monna-app/app.json` 中的 `ios.infoPlist.NSAppTransportSecurity` 配置正确 