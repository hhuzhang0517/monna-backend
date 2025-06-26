import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ValidationPipe } from '@nestjs/common';
import helmet from 'helmet';
import * as cors from 'cors';
import { LoggerService } from './modules/logger/logger.service';
import * as fs from 'fs';
import * as path from 'path';
import * as dotenv from 'dotenv';
import * as bodyParser from 'body-parser';

// 加载环境变量
dotenv.config();

// 获取当前环境
const NODE_ENV = process.env.NODE_ENV || 'development';
const isProduction = NODE_ENV === 'production';
console.log(`当前运行环境: ${NODE_ENV}`);

// 默认启用HTTPS，除非明确设置为"false"
const USE_HTTPS = process.env.USE_HTTPS !== 'false';

// 文件上传大小限制 - 统一设置
const MAX_FILE_SIZE = '100mb';
const JSON_LIMIT = '50mb';

async function bootstrap() {
  // 检查已经运行的服务并尝试终止
  try {
    console.log('检查端口占用情况...');
    const { execSync } = require('child_process');
    const output = execSync('netstat -ano | findstr :3000').toString();
    
    if (output) {
      console.log('端口3000已被占用，正在尝试释放...');
      const lines = output.split('\n');
      const pidMatches = lines.map(line => {
        const match = line.match(/\s+(\d+)\s*$/);
        return match ? match[1] : null;
      }).filter(Boolean);
      
      if (pidMatches.length > 0) {
        const pid = pidMatches[0];
        try {
          console.log(`尝试终止进程PID: ${pid}`);
          execSync(`taskkill /F /PID ${pid}`);
          console.log(`已终止进程PID: ${pid}`);
          // 等待系统释放端口
          await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (killError) {
          console.error('无法终止进程:', killError.message);
        }
      }
    }
  } catch (error) {
    console.log('端口检查失败:', error.message);
  }

  let app;
  
  // SSL证书配置
  if (USE_HTTPS) {
    // 使用环境变量或默认路径 - 更新为新的证书位置
    const SSL_CERT_PATH = process.env.SSL_CERT_PATH || path.resolve('./ssl/certificate.crt');
    const SSL_KEY_PATH = process.env.SSL_KEY_PATH || path.resolve('./ssl/server.key');
    
    console.log('启用HTTPS配置');
    console.log('SSL证书路径:', SSL_CERT_PATH);
    console.log('SSL密钥路径:', SSL_KEY_PATH);
    
    try {
      // 检查证书文件是否存在且可读
      if (!fs.existsSync(SSL_CERT_PATH)) {
        throw new Error(`证书文件不存在: ${SSL_CERT_PATH}`);
      }
      
      if (!fs.existsSync(SSL_KEY_PATH)) {
        throw new Error(`密钥文件不存在: ${SSL_KEY_PATH}`);
      }
      
      const httpsOptions = {
        key: fs.readFileSync(SSL_KEY_PATH),
        cert: fs.readFileSync(SSL_CERT_PATH),
      };
      
      // 创建带SSL的应用实例
      app = await NestFactory.create(AppModule, {
        bufferLogs: true,
        logger: new LoggerService(),
        httpsOptions,
      });
      
      // 配置请求体大小限制 - 增加上传文件大小限制
      app.use(bodyParser.json({ limit: JSON_LIMIT }));
      app.use(bodyParser.urlencoded({ limit: MAX_FILE_SIZE, extended: true }));
      // 添加原始body解析器，处理multipart
      app.use(bodyParser.raw({ limit: MAX_FILE_SIZE }));
      app.use(bodyParser.text({ limit: MAX_FILE_SIZE }));
      
      console.log('已成功启用HTTPS模式');
      console.log(`已配置请求体大小限制: JSON=${JSON_LIMIT}, 文件上传=${MAX_FILE_SIZE}`);
      
      // 同时也创建一个HTTP服务器以便Android连接
      const httpApp = await NestFactory.create(AppModule, {
        bufferLogs: true,
        logger: new LoggerService(),
      });
      
      // 为HTTP服务也配置请求体大小限制
      httpApp.use(bodyParser.json({ limit: JSON_LIMIT }));
      httpApp.use(bodyParser.urlencoded({ limit: MAX_FILE_SIZE, extended: true }));
      httpApp.use(bodyParser.raw({ limit: MAX_FILE_SIZE }));
      httpApp.use(bodyParser.text({ limit: MAX_FILE_SIZE }));
      
      httpApp.setGlobalPrefix('api');
      httpApp.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));
      
      // 安全配置 - 根据环境调整
      const helmetOptions = {
        contentSecurityPolicy: isProduction, // 生产环境启用
        crossOriginEmbedderPolicy: isProduction, // 生产环境启用
        xssFilter: true,
        noSniff: true,
        hidePoweredBy: true,
      };
      
      httpApp.use(helmet(helmetOptions));
      
      // CORS配置 - 针对移动应用放宽限制
      const corsOptions = {
        origin: isProduction ? ['https://monna.app', /\.monna\.app$/] : true, // 开发环境允许所有源
        methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
        allowedHeaders: 'Content-Type,Authorization',
        exposedHeaders: 'Content-Disposition',  // 允许客户端读取文件下载头
        preflightContinue: false,
        optionsSuccessStatus: 204,
        credentials: true,
      };
      httpApp.use(cors(corsOptions));
      
      // 启动HTTP服务
      const HTTP_PORT = parseInt(process.env.PORT || '3000') + 1; // 使用3001端口避免冲突
      httpApp.listen(HTTP_PORT, '0.0.0.0');
      console.log(`HTTP服务已启动（用于Android设备）: http://localhost:${HTTP_PORT}/api/docs`);
    } catch (error) {
      console.error('SSL证书加载失败，将回退到HTTP模式:', error.message);
      // 证书加载失败，回退到HTTP模式
      app = await NestFactory.create(AppModule, {
        bufferLogs: true,
        logger: new LoggerService(),
      });
      
      // 配置请求体大小限制
      app.use(bodyParser.json({ limit: JSON_LIMIT }));
      app.use(bodyParser.urlencoded({ limit: MAX_FILE_SIZE, extended: true }));
      app.use(bodyParser.raw({ limit: MAX_FILE_SIZE }));
      app.use(bodyParser.text({ limit: MAX_FILE_SIZE }));
      
      console.log('使用HTTP模式运行 (SSL加载失败)');
      console.log(`已配置请求体大小限制: JSON=${JSON_LIMIT}, 文件上传=${MAX_FILE_SIZE}`);
    }
  } else {
    // 创建HTTP应用实例
    app = await NestFactory.create(AppModule, {
      bufferLogs: true,
      logger: new LoggerService(),
    });
    
    // 配置请求体大小限制
    app.use(bodyParser.json({ limit: JSON_LIMIT }));
    app.use(bodyParser.urlencoded({ limit: MAX_FILE_SIZE, extended: true }));
    app.use(bodyParser.raw({ limit: MAX_FILE_SIZE }));
    app.use(bodyParser.text({ limit: MAX_FILE_SIZE }));
    
    console.log('使用HTTP模式运行 (由配置指定)');
    console.log(`已配置请求体大小限制: JSON=${JSON_LIMIT}, 文件上传=${MAX_FILE_SIZE}`);
  }
  
  app.setGlobalPrefix('api');
  app.useGlobalPipes(new ValidationPipe({ whitelist: true, transform: true }));
  
  // 安全配置 - 根据环境调整
  const helmetOptions = {
    contentSecurityPolicy: isProduction, // 生产环境启用
    crossOriginEmbedderPolicy: isProduction, // 生产环境启用
    xssFilter: true,
    noSniff: true,
    hidePoweredBy: true,
  };
  
  app.use(helmet(helmetOptions));
  
  // CORS配置 - 针对移动应用放宽限制
  const corsOptions = {
    origin: isProduction ? ['https://monna.app', /\.monna\.app$/] : true, // 开发环境允许所有源
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
    allowedHeaders: 'Content-Type,Authorization',
    exposedHeaders: 'Content-Disposition',  // 允许客户端读取文件下载头
    preflightContinue: false,
    optionsSuccessStatus: 204,
    credentials: true,
  };
  app.use(cors(corsOptions));

  // API文档配置
  const config = new DocumentBuilder()
    .setTitle('AI Photo & Video Generation API')
    .setDescription('RESTful API for AI photo/video generation, OAuth, tasks, admin, etc.')
    .setVersion('1.0')
    .addBearerAuth()
    .build();
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api/docs', app, document);

  // 在所有网络接口上监听
  const PORT = process.env.PORT || 3000;
  await app.listen(PORT, '0.0.0.0');
  
  // 日志输出
  const protocol = USE_HTTPS ? 'https' : 'http';
  console.log(`后端服务已启动：${protocol}://localhost:${PORT}/api/docs`);
  console.log(`对外访问地址：${protocol}://${process.env.SERVER_HOST || 'localhost'}:${PORT}/api/docs`);
}

bootstrap();
