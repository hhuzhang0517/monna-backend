# 后端结构说明

本项目后端采用分层微服务架构：

- **Node.js API层（NestJS）**：负责所有RESTful API、用户鉴权、任务调度、文件上传/下载、任务入队（Redis）、状态查询、Admin统计、日志采集与输出、OpenAPI接口文档。
- **Python AI Worker**：负责消费Redis任务队列，调用AI模型进行推理，写入结果文件，回写MongoDB任务状态。
- **Redis**：任务队列，异步解耦API与AI Worker。
- **MongoDB**：存储用户、任务、统计等元数据。
- **临时文件存储**：/uploads/<taskId>、/outputs/<taskId>，定时清理。

## Node.js API层（NestJS）

- 目录：`monna-backend/node-api`
- 主要职责：
  - 用户鉴权（Apple/Google OAuth、游客、JWT）
  - 任务创建（写真/视频）、文件上传、任务入队（Redis）
  - 任务状态查询、结果下载、删除
  - Admin统计接口
  - 日志采集与输出（结构化JSON日志，traceId链路追踪）
  - OpenAPI/Swagger接口文档自动生成
- 关键依赖：
  - nestjs、nestjs/swagger、mongoose、ioredis、multer、jsonwebtoken、winston（日志）、helmet、class-validator

## Python AI Worker

- 目录：`monna-backend/ai-worker`
- 主要职责：
  - 监听Redis队列，消费任务
  - 调用AI模型，写结果文件
  - 更新MongoDB任务状态
  - 结果文件清理

## 日志采集与追踪

- Node.js API层所有请求、任务、异常、关键操作均输出结构化JSON日志，字段包括：timestamp、level、service、region、taskId、userId、traceId、msg。
- traceId贯穿API、Redis、Python Worker，便于全链路追踪。
- 日志可对接Fluent Bit、ELK、Kibana等平台。
- 支持日志等级（info、warn、error、debug），异常自动记录堆栈。

## OpenAPI/Swagger接口文档

- 所有RESTful API自动生成Swagger文档，访问路径：`/api/docs`。
- 支持接口分组、参数/响应示例、鉴权说明。

## 目录结构示例

```
monna-backend/
  ├─ node-api/           # Node.js (NestJS) API服务
  ├─ ai-worker/          # Python AI推理服务
  ├─ shared/             # 公共类型、协议、文档
  ├─ docker/             # Docker部署相关
  ├─ README.md           # 后端说明
  └─ ...（现有内容保留）
```

---

# 详细API接口文档

详见`node-api`目录下的Swagger自动生成文档。

---

# 部署与运维

详见主README.md第6节。

---

# Monna 后端服务

## 服务组件

该后端系统由以下服务组成：

1. FastAPI服务 (Python) - 提供FaceChain AI人像生成、背景处理等功能
   - 监听端口: 8000
   - API基础路径: `/api/v1`

2. Node.js API服务 - 提供用户验证、任务管理等功能
   - 监听端口: 3001
   - API基础路径: `/api`

## 服务启动

1. 启动FaceChain服务:
   ```
   ./start-facechain-service.ps1
   ```

2. 启动Node.js API服务:
   ```
   ./run-dev.ps1
   ```

## 前端连接测试修复

如果前端"测试服务器连接"功能出现错误，请参考以下修复方案：

1. 后端已添加了专用的连接测试端点: `/api/v1/facechain/connection-test`

2. 前端需要修改两处代码:
   - 修复 `EnvironmentSwitcher.js` 中的 `startsWith` 错误
   - 更新 `api.js` 中的测试URL

详细修复步骤请参考 `frontend-fix-guide.txt` 文件。

## 日志目录

系统日志文件位于 `logs` 目录下：
- `logs/facechain_api.log` - FaceChain API日志
- `logs/api.log` - 通用API日志
- `logs/style_transfer.log` - 风格转换服务日志

# FaceChain 后端服务

## 快速启动指南

1. 确保安装了Python 3.10虚拟环境：

```powershell
# 如果没有创建虚拟环境
python -m venv venv310

# 激活虚拟环境
.\venv310\Scripts\Activate.ps1
```

2. 启动FaceChain服务:

```powershell
# 启动主FaceChain服务
.\start-facechain-service.ps1
```

3. 启动前端兼容代理服务器:

```powershell
# 启动代理服务器（监听8080端口）
.\start-proxy.ps1
```

4. 更新前端配置指向代理服务器:

```powershell
# 将前端API端口从8000更改为8080
.\update-frontend-config.ps1
```

5. 启动Node.js后端API服务:

```powershell
# 启动Node.js API服务
.\run-dev.ps1
```

## 前端连接问题修复

这里提供了两种解决前端连接问题的方法：

### 方法1: 使用代理服务器（推荐）

使用代理服务器可以避免修改前端代码，只需要将前端请求的端口从8000改为8080。代理服务器会自动：
- 接收发送到 `/api/v1/facechain/tasks/test-id` 的请求
- 将请求转发到FaceChain服务的 `/api/v1/facechain/connection-test` 端点
- 修复响应中缺少的 `apiBaseUrl` 字段，解决前端的 `startsWith` 错误

详细说明请查看 [proxy-fix-readme.md](./proxy-fix-readme.md)

### 方法2: 修改前端代码

如果需要直接修改前端代码，请参考:
- 更新 `EnvironmentSwitcher.js` 中的 `startsWith` 代码
- 更新 `api.js` 中的测试端点URL

## 开发指南

### 服务端口

- FaceChain服务 (Python FastAPI): 8000
- 前端兼容代理服务器: 8080
- Node.js API服务: 3001

### 日志文件

- FaceChain API: `logs/facechain_api.log`
- 代理服务器: `logs/proxy_server.log`
- API服务: `logs/api.log`

### 主要文件

- `app/api/endpoints/facechain.py`: FaceChain API接口
- `proxy-server.py`: 前端兼容代理服务器
- `start-facechain-service.ps1`: 启动FaceChain服务脚本
- `start-proxy.ps1`: 启动代理服务器脚本
- `update-frontend-config.ps1`: 更新前端配置脚本 