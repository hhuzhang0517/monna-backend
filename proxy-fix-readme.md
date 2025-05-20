# FaceChain 前端连接测试修复方案

## 问题说明

前端App尝试访问 `/api/v1/facechain/tasks/test-id` 来测试与FaceChain服务的连接，但有两个问题:

1. 该路径与现有的参数化路由 `/tasks/{task_id}` 冲突，导致404错误
2. 前端EnvironmentSwitcher组件尝试访问 `testResults.apiBaseUrl.startsWith()` 但该属性不存在

## 解决方案: 使用代理服务器

为了避免修改前端代码，我们创建了一个代理服务器，它会:

1. 监听 `/api/v1/facechain/tasks/test-id` 请求
2. 将请求代理到 `/api/v1/facechain/connection-test` 端点
3. 在响应中添加必要的 `apiBaseUrl` 字段以修复前端错误

## 使用方法

### 步骤1: 启动FaceChain后端服务

```powershell
./start-facechain-service.ps1
```

### 步骤2: 启动代理服务器

```powershell
./start-proxy.ps1
```

### 步骤3: 修改前端请求配置

修改前端API请求配置，将FaceChain API端口从8000改为8080:

```javascript
// 前端app中的配置文件（例如.env或config.js）
PYTHON_FACECHAIN_DEV_PORT=8080  // 原来是8000
```

### 验证连接

前端App可以正常连接到代理服务器，不会再有404错误，且所有测试请求都能正确处理。

## 技术细节

- 代理服务器运行在端口8080
- FaceChain后端服务运行在端口8000
- 代理服务器使用FastAPI实现
- 日志文件位于 `logs/proxy_server.log` 