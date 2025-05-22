import { Injectable, Logger, OnModuleDestroy, OnModuleInit } from '@nestjs/common';
import Redis from 'ioredis';

@Injectable()
export class RedisService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(RedisService.name);
  private redisClient: Redis;
  
  // Queue configuration
  private readonly TASK_QUEUE = 'ai_tasks';
  private readonly TASK_STATUS_CHANNEL = 'task_status';
  
  constructor() {
    // Initialize Redis client with connection parameters
    this.redisClient = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379', 10),
      db: parseInt(process.env.REDIS_DB || '0', 10),
      // Optional connection retries
      retryStrategy: (times) => {
        return Math.min(times * 100, 3000); // Increasing retry delay, max 3s
      },
    });

    // Setup event handlers
    this.redisClient.on('connect', () => {
      this.logger.log('Connected to Redis');
    });

    this.redisClient.on('error', (err) => {
      this.logger.error(`Redis connection error: ${err.message}`);
    });
  }

  /**
   * Handle module initialization 
   */
  async onModuleInit() {
    try {
      // Test connection
      await this.redisClient.ping();
      this.logger.log('Redis connection initialized');
      
      // Subscribe to task status updates
      this.setupTaskStatusSubscriber();
    } catch (error) {
      this.logger.error(`Failed to initialize Redis connection: ${error.message}`);
    }
  }

  /**
   * Handle module shutdown - close Redis connections
   */
  async onModuleDestroy() {
    // Clean up Redis connection
    if (this.redisClient) {
      this.logger.log('Closing Redis connection');
      await this.redisClient.quit();
    }
  }

  /**
   * Push AI task to Redis queue
   * @param taskData Task data object
   * @returns boolean indicating success
   */
  async pushTaskToQueue(taskData: any): Promise<boolean> {
    try {
      // Ensure taskId exists
      if (!taskData.taskId) {
        this.logger.error('Cannot push task without taskId');
        return false;
      }

      // Add timestamp if not exists
      if (!taskData.createdAt) {
        taskData.createdAt = new Date().toISOString();
      }

      // Serialize and push to queue
      const serialized = JSON.stringify(taskData);
      await this.redisClient.rpush(this.TASK_QUEUE, serialized);
      
      this.logger.log(`Task ${taskData.taskId} pushed to queue`);
      return true;
    } catch (error) {
      this.logger.error(`Failed to push task to queue: ${error.message}`);
      return false;
    }
  }

  /**
   * Subscribe to task status updates channel
   */
  private setupTaskStatusSubscriber() {
    const subscriber = this.redisClient.duplicate();
    
    subscriber.on('message', (channel, message) => {
      if (channel === this.TASK_STATUS_CHANNEL) {
        try {
          const statusUpdate = JSON.parse(message);
          this.logger.log(`Received status update for task ${statusUpdate.taskId}: ${statusUpdate.status}`);
          
          // Handle the task status update
          // This could emit events or call other services
          this.handleTaskStatusUpdate(statusUpdate);
        } catch (error) {
          this.logger.error(`Error processing task status update: ${error.message}`);
        }
      }
    });
    
    subscriber.subscribe(this.TASK_STATUS_CHANNEL, (err, count) => {
      if (err) {
        this.logger.error(`Failed to subscribe to ${this.TASK_STATUS_CHANNEL}: ${err.message}`);
      } else {
        this.logger.log(`Subscribed to ${count} channels including ${this.TASK_STATUS_CHANNEL}`);
      }
    });
  }

  /**
   * Handle task status updates
   * Implement custom logic here to react to task status changes
   */
  private handleTaskStatusUpdate(statusUpdate: any) {
    // This could update in-memory task status cache, emit events, or trigger other actions
    // For now, we just log it
    this.logger.debug(`Processing task status update: ${JSON.stringify(statusUpdate)}`);
    
    // You can emit events or call other services here
    // For example: this.eventEmitter.emit('task.statusChanged', statusUpdate);
  }
  
  /**
   * Get the Redis client instance
   */
  getClient(): Redis {
    return this.redisClient;
  }
} 