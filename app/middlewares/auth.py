from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Callable, Any
import jwt
from jwt.exceptions import PyJWTError
from app.core.config import settings

security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    """
    认证中间件，验证API请求是否包含有效的JWT令牌
    """
    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        # 只验证API路由
        if not request.url.path.startswith(settings.API_V1_STR):
            return await call_next(request)
            
        # 公开路由无需验证
        public_routes = [
            # 可以在这里添加不需要认证的路由
            # 例如: "/api/v1/auth/login", "/api/v1/auth/register"
        ]
        
        for route in public_routes:
            if request.url.path.startswith(route):
                return await call_next(request)
        
        # 获取Authorization头
        authorization: str = request.headers.get("Authorization")
        
        if not authorization:
            raise HTTPException(
                status_code=401,
                detail="未提供认证凭据"
            )
            
        try:
            # 解析Bearer令牌
            scheme, token = authorization.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail="无效的认证方案"
                )
                
            # 验证令牌
            payload = self._verify_token(token)
            
            # 将用户信息添加到请求状态中
            request.state.user = payload
            
        except PyJWTError:
            raise HTTPException(
                status_code=401,
                detail="无效的认证令牌"
            )
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail=f"认证失败: {str(e)}"
            )
            
        # 继续处理请求
        return await call_next(request)
        
    def _verify_token(self, token: str) -> dict:
        """验证JWT令牌并返回载荷"""
        try:
            # 解码令牌
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY,
                algorithms=["HS256"]
            )
            return payload
        except:
            raise PyJWTError("令牌验证失败") 