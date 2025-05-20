import { Injectable, NotFoundException, Logger } from '@nestjs/common';
import { Response } from 'express';
import * as fs from 'fs';
import * as path from 'path';
import { getOutputDir, getFilesInDir, removeUploadDir, removeOutputDir } from '../generate/file-storage.util';
import * as archiver from 'archiver';
import { RedisService } from '../shared/redis.service';

@Injectable()
export class TasksService {
  private readonly logger = new Logger(TasksService.name);
  private readonly taskStatuses = new Map<string, any>(); // In-memory cache for task statuses

  constructor(private readonly redisService: RedisService) {
    // Subscribe to task status updates
    this.setupTaskStatusSubscriber();
  }

  /**
   * 获取任务状态和进度
   * @param id 任务ID
   * @returns 任务状态和进度信息
   */
  async getStatus(id: string) {
    // 首先检查内存缓存中是否有状态
    if (this.taskStatuses.has(id)) {
      return this.taskStatuses.get(id);
    }

    // 如果没有缓存状态，尝试从文件系统检测
    const outputDir = getOutputDir(id);
    
    // 检查输出目录是否存在
    if (fs.existsSync(outputDir)) {
      const files = getFilesInDir(outputDir);
      
      if (files.length > 0) {
        // 如果有结果文件，认为任务已完成
        const status = {
          status: 'finished',
          result: {
            files: files.map(file => {
              // 确保file是字符串类型的文件路径
              const filePath = typeof file === 'string' ? file : file.path;
              const fileName = path.basename(filePath);
              const stats = fs.statSync(filePath);
              
              return {
                id: fileName,
                size: stats.size,
                downloadUrl: `/api/tasks/${id}/result/${fileName}`
              };
            })
          }
        };
        
        // 更新缓存
        this.taskStatuses.set(id, status);
        return status;
      }
    }
    
    // 默认返回一个状态未知的结果
    return {
      status: 'unknown',
      message: '任务状态未知或任务ID不存在'
    };
  }

  /**
   * 下载任务结果
   * @param id 任务ID
   * @param res Express Response对象
   */
  async downloadResults(id: string, res: Response) {
    const outputDir = getOutputDir(id);
    
    // 检查输出目录是否存在
    if (!fs.existsSync(outputDir)) {
      throw new NotFoundException(`未找到任务 ${id} 的结果文件`);
    }
    
    const files = getFilesInDir(outputDir);
    
    if (files.length === 0) {
      throw new NotFoundException(`任务 ${id} 没有结果文件`);
    }
    
    // 创建ZIP文件
    const archive = archiver('zip', {
      zlib: { level: 9 } // 最高压缩级别
    });
    
    // 设置响应头
    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename=results-${id}.zip`);
    
    // Pipe归档到响应
    archive.pipe(res);
    
    // 将所有文件添加到ZIP
    for (const file of files) {
      // 处理不同类型的文件对象
      const filePath = typeof file === 'string' ? file : file.path;
      const fileName = path.basename(filePath);
      archive.file(filePath, { name: fileName });
    }
    
    // 完成ZIP并返回
    await archive.finalize();
  }

  /**
   * 下载单个结果文件
   * @param id 任务ID
   * @param fileName 文件名
   * @param res Express Response对象
   */
  async downloadFile(id: string, fileName: string, res: Response) {
    const outputDir = getOutputDir(id);
    const filePath = path.join(outputDir, fileName);
    
    // 检查文件是否存在
    if (!fs.existsSync(filePath)) {
      throw new NotFoundException(`未找到文件 ${fileName}`);
    }
    
    // 获取文件MIME类型
    const mimeType = this.getMimeType(fileName);
    
    // 设置响应头
    res.setHeader('Content-Type', mimeType);
    res.setHeader('Content-Disposition', `inline; filename=${fileName}`);
    
    // 创建读取流并输送到响应
    const fileStream = fs.createReadStream(filePath);
    fileStream.pipe(res);
  }

  /**
   * 删除任务相关的所有数据
   * @param id 任务ID
   */
  async deleteTask(id: string) {
    // 删除上传目录
    await removeUploadDir(id);
    
    // 删除输出目录
    await removeOutputDir(id);
    
    // 移除缓存中的状态
    this.taskStatuses.delete(id);
    
    return { success: true, message: `任务 ${id} 及其相关文件已删除` };
  }

  /**
   * 根据文件扩展名确定MIME类型
   * @param fileName 文件名
   * @returns MIME类型字符串
   */
  private getMimeType(fileName: string): string {
    const ext = path.extname(fileName).toLowerCase();
    
    switch (ext) {
      case '.jpg':
      case '.jpeg':
        return 'image/jpeg';
      case '.png':
        return 'image/png';
      case '.gif':
        return 'image/gif';
      case '.mp4':
        return 'video/mp4';
      case '.zip':
        return 'application/zip';
      default:
        return 'application/octet-stream';
    }
  }

  /**
   * 设置任务状态订阅器
   * 监听来自Redis的任务状态更新
   */
  private setupTaskStatusSubscriber() {
    // 使用Redis服务的事件处理机制更新任务状态缓存
    // 将来可以扩展为使用事件触发器模式
    
    // 这部分目前由Redis服务通过Pub/Sub机制处理
    // 当有状态更新时，更新内存中的任务状态缓存
    this.redisService.getClient().duplicate().on('message', (channel, message) => {
      try {
        const statusUpdate = JSON.parse(message);
        const { taskId, status } = statusUpdate;
        
        if (taskId && status) {
          this.logger.log(`更新任务 ${taskId} 状态: ${status}`);
          this.taskStatuses.set(taskId, statusUpdate);
        }
      } catch (error) {
        this.logger.error(`处理任务状态更新失败: ${error.message}`);
      }
    });
  }
} 