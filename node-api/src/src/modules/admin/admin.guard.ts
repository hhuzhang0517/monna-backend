import { Injectable, CanActivate, ExecutionContext, UnauthorizedException } from '@nestjs/common';
import { Observable } from 'rxjs';

@Injectable()
export class AdminGuard implements CanActivate {
  canActivate(
    context: ExecutionContext,
  ): boolean | Promise<boolean> | Observable<boolean> {
    const request = context.switchToHttp().getRequest();
    
    // TODO: 通过JWT解析用户信息，验证用户role是否为admin
    // 模拟实现，后续需要通过JWT中间件注入user信息
    
    // 模拟admin用户
    const isAdmin = request.headers['x-admin-token'] === 'admin-secret-token';
    
    if (!isAdmin) {
      throw new UnauthorizedException('需要管理员权限');
    }
    
    return true;
  }
} 