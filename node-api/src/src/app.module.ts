import { Module } from '@nestjs/common';
import { AuthModule } from './modules/auth/auth.module';
import { TasksModule } from './modules/tasks/tasks.module';
import { GenerateModule } from './modules/generate/generate.module';
import { AdminModule } from './modules/admin/admin.module';
import { LoggerModule } from './modules/logger/logger.module';
import { SharedModule } from './modules/shared/shared.module';

@Module({
  imports: [
    SharedModule,
    AuthModule,
    TasksModule,
    GenerateModule,
    AdminModule,
    LoggerModule,
  ],
})
export class AppModule {}
