# AI照片生成Worker

这个模块是Monna项目的AI照片生成部分，负责从Redis任务队列获取任务，使用FaceChain生成AI人像照片，并将结果保存到指定目录。

## 功能特点

- 从Redis任务队列获取任务
- 集成[FaceChain](https://github.com/modelscope/facechain)项目生成高质量AI人像
- 支持多种风格的照片生成
- 自动下载和配置必要的模型
- 结果保存在指定任务ID目录下
- 支持任务状态更新和通知

## 目录结构

```
monna-backend/
├── models/                      # FaceChain模型目录
│   └── facechain/              # FaceChain代码(自动下载)
├── data/
│   ├── uploads/                # 上传的照片目录
│   └── outputs/                # 生成的AI照片目录(按任务ID分目录)
└── python-worker/
    ├── worker.py               # 主工作器脚本
    ├── facechain_integration.py# FaceChain集成模块
    ├── requirements.txt        # 依赖列表
    ├── start-worker.ps1        # 启动脚本
    └── README.md               # 本文档
```

## 使用方法

1. 确保已安装Python 3.8+和Redis服务器
2. 在`monna-backend/python-worker`目录下运行:

```
./start-worker.ps1
```

启动脚本将:
- 检查并创建必要的目录
- 安装所需的Python依赖
- 启动Worker监听Redis任务队列

## Redis配置

Worker默认连接到本地Redis服务器(localhost:6379)，使用以下配置:
- 队列名称: `ai_tasks`
- 任务完成通知频道: `task_complete`

可以通过环境变量修改这些配置:
- `REDIS_HOST`: Redis服务器地址
- `REDIS_PORT`: Redis服务器端口
- `REDIS_DB`: Redis数据库索引
- `REDIS_QUEUE`: 任务队列名称
- `REDIS_CHANNEL`: 任务完成通知频道

## 任务格式

Worker期望接收以下格式的任务JSON:

```json
{
  "taskId": "uuid-task-identifier",
  "files": ["path/to/photo1.jpg", "path/to/photo2.jpg", ...],
  "styles": ["portrait", "anime", ...]
}
```

## 输出格式

任务完成后，Worker会发布以下格式的结果到Redis:

```json
{
  "taskId": "uuid-task-identifier",
  "status": "finished",
  "updatedAt": "2025-05-05T10:30:00.000Z",
  "result": {
    "files": [
      {
        "id": "result_1.jpg",
        "path": "monna-backend/data/outputs/uuid-task-identifier/result_1.jpg",
        "size": 123456,
        "url": "/api/tasks/uuid-task-identifier/result/result_1.jpg"
      },
      ...
    ]
  }
}
```

## 依赖项目

- [FaceChain](https://github.com/modelscope/facechain): AI人像生成核心技术 