import { Injectable } from '@nestjs/common';

@Injectable()
export class AuthService {
  async login(body: any) {
    // TODO: Apple/Google OAuth校验，生成/更新用户，签发JWT
    return {
      token: 'mock-jwt-token',
      user: {
        id: 'mock-user-id',
        name: 'Mock User',
        role: 'user',
        isPaid: false,
      },
    };
  }

  async guest() {
    // TODO: 生成游客账号，签发JWT
    return {
      token: 'mock-guest-jwt-token',
      user: {
        id: 'mock-guest-id',
        name: 'Guest',
        role: 'guest',
        isPaid: false,
      },
    };
  }
} 