import { Injectable, Logger, InternalServerErrorException } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid';
import * as path from 'path';
import { getUploadDir, getFilesInDir } from './file-storage.util';
import { RedisService } from '../shared/redis.service';

@Injectable()
export class GenerateService {
  private readonly logger = new Logger(GenerateService.name);

  constructor(private readonly redisService: RedisService) {}

  /**
   * 处理写真生成任务
   * @param files 用户上传的照片文件数组
   * @param body 包含styles风格数组和taskId
   * @returns 任务ID和状态
   */
  async generatePhoto(files: Express.Multer.File[], body: any) {
    const taskId = body.taskId || uuidv4();
    const styles = Array.isArray(body.styles) ? body.styles : [];
    
    // 获取所有已上传的图片路径信息
    const uploadDir = getUploadDir(taskId);
    const uploadedFiles = getFilesInDir(uploadDir);
    
    this.logger.log(`创建写真任务: ${taskId}, 图片数: ${uploadedFiles.length}, 风格: ${styles.join(', ')}`);
    
    // 准备任务数据
    const taskData = {
      taskId,
      taskType: 'photo',
      styles,
      // 添加照片路径 - 将上传的文件路径转换为相对于worker的路径
      filePaths: uploadedFiles.map(file => file.path),
      photos: uploadedFiles,
      createdAt: new Date().toISOString(),
    };
    
    // 发送任务到Redis队列
    const queued = await this.redisService.pushTaskToQueue(taskData);
    
    if (!queued) {
      this.logger.error(`将写真任务 ${taskId} 发送到Redis队列失败`);
      throw new InternalServerErrorException('无法将任务加入处理队列，请稍后再试');
    }
    
    this.logger.log(`写真任务 ${taskId} 已成功加入队列`);
    
    // 返回任务信息
    return {
      taskId,
      status: 'queued',
      message: '任务已加入队列，正在处理...'
    };
  }

  /**
   * 处理视频特效任务
   * @param files 用户上传的自拍照片
   * @param body 包含effectType特效类型和taskId
   * @returns 任务ID和状态
   */
  async generateVideo(files: Express.Multer.File[], body: any) {
    const taskId = body.taskId || uuidv4();
    const effectType = body.effectType;
    
    // 获取已上传的自拍照片路径
    const uploadDir = getUploadDir(taskId);
    const uploadedFiles = getFilesInDir(uploadDir);
    const selfieFile = uploadedFiles.length > 0 ? uploadedFiles[0] : null;
    
    if (!selfieFile) {
      throw new Error('自拍照片上传失败');
    }
    
    this.logger.log(`创建视频特效任务: ${taskId}, 特效类型: ${effectType}, 照片: ${selfieFile.name}`);
    
    // 准备任务数据
    const taskData = {
      taskId,
      taskType: 'video',
      effectType,
      // 添加照片路径
      filePath: selfieFile.path,
      selfie: selfieFile,
      createdAt: new Date().toISOString(),
    };
    
    // 发送任务到Redis队列
    const queued = await this.redisService.pushTaskToQueue(taskData);
    
    if (!queued) {
      this.logger.error(`将视频任务 ${taskId} 发送到Redis队列失败`);
      throw new InternalServerErrorException('无法将任务加入处理队列，请稍后再试');
    }
    
    this.logger.log(`视频任务 ${taskId} 已成功加入队列`);
    
    // 返回任务信息
    return {
      taskId,
      status: 'queued',
      message: '任务已加入队列，正在处理...'
    };
  }
} 