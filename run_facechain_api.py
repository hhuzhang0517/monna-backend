import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import sys

# 导入FaceChain端点
from app.api.endpoints.facechain import router as facechain_router

# 创建独立的FastAPI应用
app = FastAPI(title="Monna FaceChain API", description="FaceChain AI人像生成服务")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 创建API路由
api_router = APIRouter(prefix="/api/v1")

# 包含FaceChain路由
api_router.include_router(facechain_router, prefix="/facechain", tags=["facechain"])

# 将路由添加到应用
app.include_router(api_router)

# 设置静态文件目录
base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"
# 确保目录存在
os.makedirs(data_dir / "outputs", exist_ok=True)

# 挂载静态文件目录
app.mount("/api/v1/static", StaticFiles(directory=str(data_dir)), name="static")

if __name__ == "__main__":
    # 在开发模式下启动，开启重载功能
    uvicorn.run("run_facechain_api:app", host="0.0.0.0", port=8000, reload=True)
 