from celery import Celery
from app.core.config import settings

# 创建Celery实例
celery_app = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.worker.tasks"]
)

# 任务相关配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=False,
    result_expires=60 * 60 * 24,  # 结果过期时间(1天)
    task_track_started=True,
    worker_max_tasks_per_child=200,  # 每个worker处理的最大任务数
    task_soft_time_limit=300,  # 软超时(5分钟)
    task_time_limit=600,  # 硬超时(10分钟)
)