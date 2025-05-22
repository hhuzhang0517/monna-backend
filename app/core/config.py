import os
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from pathlib import Path

# 基础目录
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 环境变量配置
class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI修图后端服务"
    
    # 基础目录
    BASE_DIR: Path = BASE_DIR
    
    # 安全配置
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS配置
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # 文件路径
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "data" / "results"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"  # 添加日志目录
    
    # 任务队列配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    CELERY_BROKER_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    CELERY_RESULT_BACKEND: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"
    
    # 图像处理参数
    MAX_IMAGE_SIZE: int = 4096  # 最大图像尺寸(像素)
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 最大文件大小(10MB)
    
    # 模型路径配置
    MODEL_PATHS: Dict[str, Path] = {
        "u2net": MODELS_DIR / "u2net" / "u2net.pth",
        "modnet": MODELS_DIR / "u2net" / "modnet.pth",
        "cartoon": MODELS_DIR / "cartoon" / "photo2cartoon.pth",
        "animegan": MODELS_DIR / "cartoon" / "animegan2.pth",
        "cp_vton": MODELS_DIR / "vton" / "cp_vton_plus.pth",
        "lama": MODELS_DIR / "lama" / "lama.pth",
        "stargan": MODELS_DIR / "stargan" / "stargan.pth",
        "aging_gan": MODELS_DIR / "aging" / "aging_gan.pth",
    }
    
    # 国际化支持
    SUPPORTED_LANGUAGES: List[str] = ["zh-CN", "en-US"]
    DEFAULT_LANGUAGE: str = "zh-CN"
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# 创建设置实例
settings = Settings()

# 确保目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.RESULTS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)  # 创建日志目录