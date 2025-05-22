import { PipeTransform, Injectable, ArgumentMetadata, BadRequestException } from '@nestjs/common';

@Injectable()
export class FileValidationPipe implements PipeTransform {
  constructor(
    private readonly minFiles: number,
    private readonly maxFiles: number,
  ) {}

  transform(files: Express.Multer.File[], metadata: ArgumentMetadata) {
    if (!files || !Array.isArray(files)) {
      throw new BadRequestException('未上传文件');
    }

    if (files.length < this.minFiles) {
      throw new BadRequestException(`至少需要上传 ${this.minFiles} 个文件`);
    }

    if (files.length > this.maxFiles) {
      throw new BadRequestException(`最多允许上传 ${this.maxFiles} 个文件`);
    }

    return files;
  }
} 