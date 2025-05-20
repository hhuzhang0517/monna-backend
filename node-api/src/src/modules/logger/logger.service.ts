import { ConsoleLogger, Injectable, Scope } from '@nestjs/common';
import * as winston from 'winston';
import { v4 as uuidv4 } from 'uuid';

@Injectable({ scope: Scope.TRANSIENT })
export class LoggerService extends ConsoleLogger {
  private logger: winston.Logger;

  constructor() {
    super();
    this.logger = winston.createLogger({
      level: 'info',
      format: winston.format.json(),
      transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
      ],
    });
  }

  log(message: string, context?: string) {
    this.logger.info({
      timestamp: new Date().toISOString(),
      level: 'info',
      service: context || 'app',
      traceId: uuidv4(),
      msg: message,
    });
    super.log(message, context);
  }

  error(message: string, trace?: string, context?: string) {
    this.logger.error({
      timestamp: new Date().toISOString(),
      level: 'error',
      service: context || 'app',
      traceId: uuidv4(),
      msg: message,
      stack: trace,
    });
    super.error(message, trace, context);
  }

  warn(message: string, context?: string) {
    this.logger.warn({
      timestamp: new Date().toISOString(),
      level: 'warn',
      service: context || 'app',
      traceId: uuidv4(),
      msg: message,
    });
    super.warn(message, context);
  }

  debug(message: string, context?: string) {
    this.logger.debug({
      timestamp: new Date().toISOString(),
      level: 'debug',
      service: context || 'app',
      traceId: uuidv4(),
      msg: message,
    });
    super.debug?.(message, context);
  }
} 