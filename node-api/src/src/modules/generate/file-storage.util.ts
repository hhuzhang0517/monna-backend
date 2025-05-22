import { diskStorage } from 'multer';
import { v4 as uuidv4 } from 'uuid';
import * as fs from 'fs';
import * as path from 'path';
import { Request } from 'express';

// 配置可按环境变量修改 - 修改默认路径为统一的data目录
const BASE_DATA_DIR = process.env.DATA_DIR || path.resolve(process.cwd(), '..', '..', 'data');
export const UPLOAD_ROOT = process.env.UPLOAD_DIR || path.join(BASE_DATA_DIR, 'uploads');
export const OUTPUT_ROOT = process.env.OUTPUT_DIR || path.join(BASE_DATA_DIR, 'outputs');
export const FILE_EXPIRE_DAYS = 7; // 结果文件保留天数

// 确保基础目录存在
(() => {
  try {
    // 创建基础数据目录
    if (!fs.existsSync(BASE_DATA_DIR)) {
      fs.mkdirSync(BASE_DATA_DIR, { recursive: true });
      console.log(`已创建基础数据目录: ${BASE_DATA_DIR}`);
    }
    
    // 创建上传目录
    if (!fs.existsSync(UPLOAD_ROOT)) {
      fs.mkdirSync(UPLOAD_ROOT, { recursive: true });
      console.log(`已创建上传文件目录: ${UPLOAD_ROOT}`);
    }
    
    // 创建输出目录
    if (!fs.existsSync(OUTPUT_ROOT)) {
      fs.mkdirSync(OUTPUT_ROOT, { recursive: true });
      console.log(`已创建输出文件目录: ${OUTPUT_ROOT}`);
    }
    
    console.log(`文件存储配置: 数据目录=${BASE_DATA_DIR}, 上传目录=${UPLOAD_ROOT}, 输出目录=${OUTPUT_ROOT}`);
  } catch (error) {
    console.error('创建文件存储目录失败:', error);
  }
})();

// 允许的图片类型
export const ALLOWED_MIME_TYPES = [
  'image/jpeg',
  'image/jpg',
  'image/png',
  'image/webp',
  'image/heic',
  'image/heif',
];

/**
 * 创建Multer存储配置
 * 按taskId分目录保存上传文件
 */
export function createMulterStorage() {
  return diskStorage({
    destination: (req: Request, file, cb) => {
      // 任务ID：每次任务生成唯一ID
      let taskId = req.body.taskId;
      if (!taskId) {
        taskId = uuidv4();
        req.body.taskId = taskId;
      }
      const dir = getUploadDir(taskId);
      fs.mkdirSync(dir, { recursive: true });
      console.log(`文件将存储在: ${dir}`);
      cb(null, dir);
    },
    filename: (req, file, cb) => {
      // 保证文件名唯一，保留原始扩展名
      const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
      const filename = `${Date.now()}-${path.basename(file.originalname, ext)}${ext}`;
      cb(null, filename);
    },
  });
}

/**
 * 文件过滤器 - 只允许图片类型
 */
export function fileFilter(req, file, cb) {
  if (!ALLOWED_MIME_TYPES.includes(file.mimetype)) {
    return cb(new Error(`不支持的文件类型: ${file.mimetype}`), false);
  }
  cb(null, true);
}

/**
 * 获取上传目录路径
 */
export function getUploadDir(taskId: string): string {
  return path.join(UPLOAD_ROOT, taskId);
}

/**
 * 获取输出目录路径
 */
export function getOutputDir(taskId: string): string {
  return path.join(OUTPUT_ROOT, taskId);
}

/**
 * 删除源文件目录
 */
export function removeUploadDir(taskId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const dir = getUploadDir(taskId);
    if (fs.existsSync(dir)) {
      fs.rm(dir, { recursive: true, force: true }, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    } else {
      resolve();
    }
  });
}

/**
 * 删除结果文件目录
 */
export function removeOutputDir(taskId: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const dir = getOutputDir(taskId);
    if (fs.existsSync(dir)) {
      fs.rm(dir, { recursive: true, force: true }, (err) => {
        if (err) {
          reject(err);
        } else {
          resolve();
        }
      });
    } else {
      resolve();
    }
  });
}

/**
 * 创建输出结果目录
 */
export function createOutputDir(taskId: string): string {
  const dir = getOutputDir(taskId);
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

/**
 * 获取目录下所有文件路径 (字符串数组形式)
 * @param dir 目录路径
 * @param returnStrings 是否返回字符串数组
 * @returns 文件路径字符串数组
 */
export function getFilesInDir(dir: string, returnStrings: true): string[];

/**
 * 获取目录下所有文件信息 (对象数组形式)
 * @param dir 目录路径
 * @param returnStrings 是否返回字符串数组
 * @returns 文件信息对象数组
 */
export function getFilesInDir(dir: string, returnStrings?: false): Array<{ path: string, size: number, name: string }>;

/**
 * 获取目录下所有文件信息 (实现函数)
 */
export function getFilesInDir(dir: string, returnStrings = false): Array<{ path: string, size: number, name: string }> | string[] {
  if (!fs.existsSync(dir)) {
    return returnStrings ? [] : [];
  }

  const files = fs.readdirSync(dir)
    .filter(file => {
      try {
        return fs.statSync(path.join(dir, file)).isFile();
      } catch (err) {
        console.error(`Error checking file ${file}:`, err);
        return false;
      }
    });

  if (returnStrings) {
    // 返回文件路径字符串数组
    return files.map(file => path.join(dir, file));
  } else {
    // 返回文件信息对象数组
    return files.map(file => {
      const filePath = path.join(dir, file);
      const stats = fs.statSync(filePath);
      return {
        path: filePath,
        name: file,
        size: stats.size,
      };
    });
  }
}

/**
 * 清理过期文件
 * 可由定时任务调用，删除超过7天的结果文件
 */
export function cleanupExpiredFiles(): void {
  const now = Date.now();
  const expireTime = now - FILE_EXPIRE_DAYS * 24 * 60 * 60 * 1000;
  
  // 确保目录存在
  if (!fs.existsSync(OUTPUT_ROOT)) return;
  
  // 遍历输出目录
  fs.readdirSync(OUTPUT_ROOT).forEach(taskId => {
    const taskDir = path.join(OUTPUT_ROOT, taskId);
    const stats = fs.statSync(taskDir);
    
    // 如果目录创建时间超过7天，删除
    if (stats.birthtimeMs < expireTime) {
      fs.rmSync(taskDir, { recursive: true, force: true });
    }
  });
} 