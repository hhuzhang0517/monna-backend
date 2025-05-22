from fastapi import Request, HTTPException
from typing import Dict, Optional, Callable, Any
import time
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings

class RateLimiter(BaseHTTPMiddleware):
    """
    请求速率限制中间件
    """
    def __init__(
        self, 
        app, 
        requests_limit: int = 100,  # 默认限制为每分钟100个请求
        time_window: int = 60       # 时间窗口为60秒（1分钟）
    ):
        super().__init__(app)
        self.requests_limit = requests_limit
        self.time_window = time_window
        self.request_counts: Dict[str, Dict[float, int]] = {}
        self.lock = asyncio.Lock()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # 获取客户端IP地址
        client_ip = request.client.host if request.client else "unknown"
        
        # API路由才进行限制
        if not request.url.path.startswith(settings.API_V1_STR):
            return await call_next(request)
        
        # 检查请求频率
        exceeded = await self._is_rate_limited(client_ip)
        if exceeded:
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试"
            )
            
        # 继续处理请求
        return await call_next(request)
        
    async def _is_rate_limited(self, client_ip: str) -> bool:
        """检查是否超出速率限制"""
        current_time = time.time()
        
        async with self.lock:
            # 初始化客户端记录
            if client_ip not in self.request_counts:
                self.request_counts[client_ip] = {}
                
            # 清理旧记录
            self._clean_old_records(client_ip, current_time)
            
            # 计算当前窗口内的请求数
            request_count = sum(self.request_counts[client_ip].values())
            
            # 检查是否超过限制
            if request_count >= self.requests_limit:
                return True
                
            # 增加请求计数
            if current_time in self.request_counts[client_ip]:
                self.request_counts[client_ip][current_time] += 1
            else:
                self.request_counts[client_ip][current_time] = 1
                
            return False
    
    def _clean_old_records(self, client_ip: str, current_time: float) -> None:
        """清理超出时间窗口的记录"""
        cutoff_time = current_time - self.time_window
        
        # 删除旧的时间戳记录
        self.request_counts[client_ip] = {
            timestamp: count 
            for timestamp, count in self.request_counts[client_ip].items() 
            if timestamp > cutoff_time
        } 