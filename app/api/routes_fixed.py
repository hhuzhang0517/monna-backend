from fastapi import APIRouter
from app.api.endpoints import background, preprocessing, facechain

api_router = APIRouter()

# 注册背景处理相关的路由
api_router.include_router(background.router, prefix="/background", tags=["背景处理"])

# 注册图像预处理相关的路由
api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["图像预处理"]) 

# 注册FaceChain AI人像生成相关的路由
api_router.include_router(facechain.router, prefix="/facechain", tags=["AI人像生成"])
