# Shared Module

这个模块包含了在多个模块之间共享的服务和工具。

## Redis Service

`RedisService` 提供了与Redis交互的功能，主要用于：

1. **任务队列管理**：
   - 将AI照片和视频生成任务推送到Redis队列
   - Python Worker从队列中获取并处理任务

2. **任务状态更新**：
   - 订阅任务完成事件通道
   - 更新本地任务状态缓存

## 使用方法

1. 导入`SharedModule`到你的模块：

```typescript
import { Module } from '@nestjs/common';
import { SharedModule } from '../shared/shared.module';

@Module({
  imports: [
    SharedModule,
  ],
  // ...
})
export class YourModule {}
```

2. 在服务中注入`RedisService`：

```typescript
import { Injectable } from '@nestjs/common';
import { RedisService } from '../shared/redis.service';

@Injectable()
export class YourService {
  constructor(private readonly redisService: RedisService) {}
  
  async someMethod() {
    // 向任务队列推送数据
    await this.redisService.pushTaskToQueue({
      taskId: 'some-uuid',
      // 其他任务数据...
    });
  }
}
```

## 配置

Redis连接参数可以通过环境变量配置：

- `REDIS_HOST`: Redis服务器地址（默认: localhost）
- `REDIS_PORT`: Redis服务器端口（默认: 6379）
- `REDIS_DB`: Redis数据库索引（默认: 0） 