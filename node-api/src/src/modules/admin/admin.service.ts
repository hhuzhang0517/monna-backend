import { Injectable, Logger } from '@nestjs/common';

// 将DauData改为导出接口
export interface DauData {
  date: string;
  dau: number;
  paid: number;
}

@Injectable()
export class AdminService {
  private readonly logger = new Logger(AdminService.name);

  /**
   * 获取今日统计概览
   * @returns 今日DAU/付费率/任务量
   */
  async getStatsOverview() {
    this.logger.log('获取今日统计概览');
    
    // TODO: 从MongoDB获取统计数据
    return {
      date: new Date().toISOString().split('T')[0],
      dau: 1234,
      paidRate: 0.15,
      taskCount: 5678,
      queueLength: 42,
      gpuUtilization: 0.85,
    };
  }

  /**
   * 获取按国家的用户分布
   * @returns 按国家用户分布JSON列表
   */
  async getStatsGeography() {
    this.logger.log('获取用户地理分布');
    
    // TODO: 从MongoDB获取地理分布数据
    return [
      { country: '中国', users: 5000, percentage: 0.45 },
      { country: '美国', users: 2500, percentage: 0.23 },
      { country: '日本', users: 1200, percentage: 0.11 },
      { country: '韩国', users: 800, percentage: 0.07 },
      { country: '其他', users: 1500, percentage: 0.14 },
    ];
  }

  /**
   * 获取DAU趋势
   * @param range 时间范围，如7d, 30d
   * @returns DAU趋势数据
   */
  async getStatsDau(range: string) {
    this.logger.log(`获取DAU趋势，范围: ${range}`);
    
    // 根据range参数决定返回数据粒度
    const days = parseInt(range.replace('d', '')) || 7;
    
    // TODO: 从MongoDB获取DAU趋势数据
    const data: DauData[] = [];
    const today = new Date();
    
    for (let i = days - 1; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(date.getDate() - i);
      
      data.push({
        date: date.toISOString().split('T')[0],
        dau: Math.floor(1000 + Math.random() * 1000),
        paid: Math.floor(100 + Math.random() * 300),
      });
    }
    
    return {
      range,
      data,
    };
  }
} 