import sys
from pathlib import Path

project_root_p_str = str(Path(__file__).resolve().parent)
facechain_base_p_str = str(Path(project_root_p_str) / "models" / "facechain")

# 先移除旧的路径（如果存在），以便控制插入顺序
if facechain_base_p_str in sys.path:
    sys.path.remove(facechain_base_p_str)
if project_root_p_str in sys.path:
    sys.path.remove(project_root_p_str)

# 插入路径，确保 project_root_p_str 在最前面 (sys.path[0])
# 这样 'app' 包可以从项目根目录正确导入
# 然后插入 facechain_base_p_str (它将成为 sys.path[1])
# 这样 'facechain' 包可以从 monna-backend/models/facechain 目录导入
sys.path.insert(0, facechain_base_p_str)
sys.path.insert(0, project_root_p_str)

# 可选的调试打印:
# from app.api.endpoints.facechain import logger as facechain_logger # 避免循环导入问题，如果需要日志
# facechain_logger.info(f"Corrected sys.path in main.py: {sys.path}")

# FastAPI 和其他应用相关的导入应该在此之后
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import api_router
from app.core.config import settings
from app.middlewares.rate_limiter import RateLimiter
# 根据需要启用或禁用认证中间件
# from app.middlewares.auth import AuthMiddleware
import os
# from pathlib import Path # Path is already imported above
import logging # 添加logging导入

# 从facechain端点导入初始化函数
from app.api.endpoints.facechain import initialize_facechain_model, logger as facechain_logger # 使用facechain的logger

# 确保数据目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.BASE_DIR / "logs", exist_ok=True)  # 确保主日志目录存在

# 确保其他可能需要的日志目录也存在
logs_dir = settings.BASE_DIR / "logs"
os.makedirs(logs_dir, exist_ok=True)

# 确保FaceChain相关目录存在
facechain_upload_dir = settings.UPLOAD_DIR / "facechain"
facechain_output_dir = settings.BASE_DIR / "data" / "outputs"
os.makedirs(facechain_upload_dir, exist_ok=True)
os.makedirs(facechain_output_dir, exist_ok=True)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI图像处理服务",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# --- FastAPI Lifespan Event Handler ---
@app.on_event("startup")
async def startup_event():
    facechain_logger.info("Application startup: Initializing FaceChain model...")
    initialize_facechain_model() # 调用模型和风格初始化
    facechain_logger.info("FaceChain model initialization attempt complete.")
    
    # 初始化任务队列系统
    from app.worker.queue import initialize_queue
    await initialize_queue()
    facechain_logger.info("In-memory task queue system initialized.")
    
    # 初始化Redis队列连接
    try:
        from app.worker.redis_queue import initialize_redis
        redis_initialized = await initialize_redis()
        if redis_initialized:
            facechain_logger.info("Redis queue connection initialized successfully.")
        else:
            facechain_logger.warning("Failed to initialize Redis queue connection. Will use in-memory queue only.")
    except Exception as e:
        facechain_logger.error(f"Error initializing Redis queue: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    facechain_logger.info("Application shutdown: Cleaning up resources...")
    # 关闭任务队列系统
    from app.worker.queue import shutdown_queue
    shutdown_queue()
    facechain_logger.info("In-memory task queue system shutdown complete.")
    
    # 关闭Redis连接
    try:
        from app.worker.redis_queue import shutdown_redis
        await shutdown_redis()
        facechain_logger.info("Redis connection shutdown complete.")
    except Exception as e:
        facechain_logger.error(f"Error shutting down Redis connection: {str(e)}")

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 添加速率限制中间件
app.add_middleware(
    RateLimiter,
    requests_limit=60,  # 每分钟60个请求
    time_window=60,     # 1分钟时间窗口
)

# 根据需要添加认证中间件
# app.add_middleware(AuthMiddleware)

# 添加API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

# 静态文件服务
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory=str(settings.BASE_DIR)), name="static")

@app.get("/")
async def root():
    return {
        "message": "欢迎使用AI修图后端服务",
        "docs": f"{settings.API_V1_STR}/docs"
    }

if __name__ == "__main__":
    import uvicorn
    # 开发环境运行配置
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 