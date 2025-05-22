import 'multer';

import { Controller, Post, UploadedFiles, Body, UseInterceptors, BadRequestException, InternalServerErrorException, Logger } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiConsumes, ApiBearerAuth, ApiBody } from '@nestjs/swagger';
import { GenerateService } from './generate.service';
import { AnyFilesInterceptor } from '@nestjs/platform-express';
import { createMulterStorage, fileFilter } from './file-storage.util';
import { FileValidationPipe } from './file-validation.pipe';

@ApiTags('generate')
@Controller('generate')
@ApiBearerAuth()
export class GenerateController {
  private readonly logger = new Logger(GenerateController.name);

  constructor(private readonly generateService: GenerateService) {}

  @Post('photo')
  @ApiOperation({ summary: '上传8-12张照片，创建写真任务' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      required: ['styles', 'photos'],
      properties: {
        styles: {
          type: 'string',
          example: '婚纱',
          description: '照片风格，如：婚纱、油画、古风、男模、漫画等'
        },
        photos: {
          type: 'array',
          items: {
            type: 'string',
            format: 'binary'
          },
          description: '照片文件，请选择8-12张照片（支持多选）'
        }
      }
    }
  })
  @UseInterceptors(AnyFilesInterceptor({
    storage: createMulterStorage(),
    fileFilter,
    limits: { 
      fileSize: 20 * 1024 * 1024, // 增加到20MB，解决大图片上传问题
      files: 12,                  // 最多12个文件
      fieldSize: 20 * 1024 * 1024 // 表单字段大小限制
    }
  }))
  @ApiResponse({ status: 200, description: '任务ID和状态' })
  async generatePhoto(
    @UploadedFiles(new FileValidationPipe(8, 12)) files: Express.Multer.File[],
    @Body() body: any,
  ) {
    try {
      if (!files || files.length < 8 || files.length > 12) {
        throw new BadRequestException('写真任务必须上传8-12张照片');
      }

      // 处理风格字段 - 转换为数组格式
      let styles: string[] = [];
      if (body.styles) {
        // 前端可能发送单个字符串或带[]的字段
        styles = [body.styles];
      } else if (body['styles[]']) {
        // 处理可能的数组风格
        styles = Array.isArray(body['styles[]']) ? body['styles[]'] : [body['styles[]']];
      }

      if (!styles || styles.length === 0) {
        throw new BadRequestException('必须选择风格');
      }

      this.logger.log(`接收到写真任务请求: ${body.taskId}, 照片数: ${files?.length}, 风格: ${styles.join(',')}`);
      
      // 确保body中包含styles数组
      const modifiedBody = { ...body, styles };
      return this.generateService.generatePhoto(files, modifiedBody);
    } catch (error) {
      this.logger.error(`写真任务创建失败: ${error.message}`, error.stack);
      throw error instanceof BadRequestException 
        ? error 
        : new InternalServerErrorException('写真任务创建失败');
    }
  }

  @Post('video')
  @ApiOperation({ summary: '上传1张自拍，创建特效视频任务' })
  @ApiConsumes('multipart/form-data')
  @ApiBody({
    schema: {
      type: 'object',
      required: ['effectType', 'selfie'],
      properties: {
        effectType: {
          type: 'string',
          example: '老照片',
          description: '视频特效类型，如：老照片、少林火云掌等'
        },
        selfie: {
          type: 'string',
          format: 'binary',
          description: '自拍照片文件'
        }
      }
    }
  })
  @UseInterceptors(AnyFilesInterceptor({
    storage: createMulterStorage(),
    fileFilter,
    limits: { 
      fileSize: 20 * 1024 * 1024, // 增加到20MB，解决大图片上传问题
      files: 1,                   // 最多1个文件
      fieldSize: 20 * 1024 * 1024 // 表单字段大小限制
    }
  }))
  @ApiResponse({ status: 200, description: '任务ID和状态' })
  async generateVideo(
    @UploadedFiles(new FileValidationPipe(1, 1)) files: Express.Multer.File[],
    @Body() body: any,
  ) {
    try {
      if (!files || files.length !== 1) {
        throw new BadRequestException('视频任务必须上传1张自拍照片');
      }

      if (!body.effectType) {
        throw new BadRequestException('必须选择视频特效类型');
      }

      this.logger.log(`接收到视频任务请求: ${body.taskId}, 特效类型: ${body.effectType}`);
      return this.generateService.generateVideo(files, body);
    } catch (error) {
      this.logger.error(`视频任务创建失败: ${error.message}`, error.stack);
      throw error instanceof BadRequestException 
        ? error 
        : new InternalServerErrorException('视频任务创建失败');
    }
  }
} 