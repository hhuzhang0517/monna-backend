#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceChain前端兼容代理服务器
用于代理前端的请求到后端正确的端点，避免修改前端代码
"""

import logging
import os
from pathlib import Path
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# 配置日志
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(logs_dir / "proxy_server.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("proxy-server")

# 创建应用
app = FastAPI(title="FaceChain前端兼容代理服务器")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 后端服务配置
FACECHAIN_API_BASE = "http://localhost:8000/api/v1/facechain"

@app.get("/api/v1/facechain/tasks/test-id")
async def proxy_test_endpoint(request: Request, response: Response):
    """
    将/api/v1/facechain/tasks/test-id请求代理到/api/v1/facechain/connection-test
    """
    logger.info("接收到前端测试请求，正在代理到connection-test端点")
    
    try:
        # 转发请求到正确的端点
        backend_response = requests.get(
            f"{FACECHAIN_API_BASE}/connection-test",
            params=dict(request.query_params),
            headers={key: value for key, value in request.headers.items() if key.lower() != 'host'}
        )
        
        # 复制响应状态码
        response.status_code = backend_response.status_code
        
        # 获取响应内容
        result = backend_response.json()
        logger.info(f"成功代理请求，状态码: {backend_response.status_code}")
        
        # 添加apiBaseUrl字段以修复前端startsWith错误
        if "apiBaseUrl" not in result:
            request_host = request.headers.get("host", "localhost:8000")
            scheme = request.headers.get("x-forwarded-proto", "http")
            api_base_url = f"{scheme}://{request_host}/api/v1"
            
            result["apiBaseUrl"] = api_base_url
            result["pythonApiBaseUrl"] = api_base_url
            result["nodeApiBaseUrl"] = f"{scheme}://{request_host.split(':')[0]}:3001/api"
        
        return result
        
    except Exception as e:
        logger.error(f"代理请求时出错: {e}")
        raise HTTPException(status_code=502, detail=f"代理请求失败: {str(e)}")

@app.get("/proxy-status")
async def proxy_status():
    """检查代理服务器状态"""
    return {
        "status": "ok",
        "message": "代理服务器运行正常",
        "proxy_path": "/api/v1/facechain/tasks/test-id -> /api/v1/facechain/connection-test"
    }

if __name__ == "__main__":
    PORT = int(os.environ.get("PROXY_PORT", 8080))
    HOST = os.environ.get("PROXY_HOST", "0.0.0.0")
    
    logger.info(f"启动前端兼容代理服务器在 {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT) 