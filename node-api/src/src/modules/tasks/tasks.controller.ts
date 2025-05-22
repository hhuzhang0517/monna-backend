import { Controller, Get, Post, Delete, Param, Res, StreamableFile, Header, HttpStatus } from '@nestjs/common';
import { ApiOperation, ApiParam, ApiResponse, ApiTags } from '@nestjs/swagger';
import { TasksService } from './tasks.service';
import { Response } from 'express';

@ApiTags('任务管理')
@Controller('tasks')
export class TasksController {
  constructor(private readonly tasksService: TasksService) {}

  @ApiOperation({ summary: '获取任务状态' })
  @ApiParam({ name: 'id', description: '任务ID' })
  @ApiResponse({ status: 200, description: '成功获取任务状态' })
  @Get(':id/status')
  async getStatus(@Param('id') id: string) {
    return this.tasksService.getStatus(id);
  }

  @ApiOperation({ summary: '下载任务结果ZIP包' })
  @ApiParam({ name: 'id', description: '任务ID' })
  @ApiResponse({ status: 200, description: '返回ZIP文件' })
  @ApiResponse({ status: 404, description: '任务结果不存在' })
  @Get(':id/results')
  async downloadResults(@Param('id') id: string, @Res() res: Response) {
    return this.tasksService.downloadResults(id, res);
  }

  @ApiOperation({ summary: '下载特定结果文件' })
  @ApiParam({ name: 'id', description: '任务ID' })
  @ApiParam({ name: 'fileName', description: '文件名' })
  @ApiResponse({ status: 200, description: '返回请求的文件' })
  @ApiResponse({ status: 404, description: '文件不存在' })
  @Get(':id/result/:fileName')
  async downloadFile(
    @Param('id') id: string,
    @Param('fileName') fileName: string,
    @Res() res: Response,
  ) {
    return this.tasksService.downloadFile(id, fileName, res);
  }

  @ApiOperation({ summary: '删除任务及相关数据' })
  @ApiParam({ name: 'id', description: '任务ID' })
  @ApiResponse({ status: 200, description: '成功删除任务' })
  @Delete(':id')
  async deleteTask(@Param('id') id: string) {
    return this.tasksService.deleteTask(id);
  }
} 