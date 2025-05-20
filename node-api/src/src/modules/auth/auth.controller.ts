import { Controller, Post, Body } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse } from '@nestjs/swagger';
import { AuthService } from './auth.service';

@ApiTags('auth')
@Controller('auth')
export class AuthController {
  constructor(private readonly authService: AuthService) {}

  @Post('login')
  @ApiOperation({ summary: 'Apple/Google 登录' })
  @ApiResponse({ status: 200, description: 'JWT和用户信息' })
  async login(@Body() body: any) {
    return this.authService.login(body);
  }

  @Post('guest')
  @ApiOperation({ summary: '游客临时账号' })
  @ApiResponse({ status: 200, description: 'JWT和游客信息' })
  async guest() {
    return this.authService.guest();
  }
} 