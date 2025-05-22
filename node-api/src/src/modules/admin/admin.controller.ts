import { Controller, Get, Query } from '@nestjs/common';
import { ApiTags, ApiOperation, ApiResponse, ApiBearerAuth } from '@nestjs/swagger';
import { AdminService, DauData } from './admin.service';

@ApiTags('admin')
@Controller('admin')
@ApiBearerAuth()
// @UseGuards(AdminGuard) // TODO: 实现并启用Admin权限守卫
export class AdminController {
  constructor(private readonly adminService: AdminService) {}

  @Get('stats/overview')
  @ApiOperation({ summary: '获取今日统计概览' })
  @ApiResponse({ status: 200, description: '今日DAU/付费率/任务量' })
  async getStatsOverview() {
    return this.adminService.getStatsOverview();
  }

  @Get('stats/geography')
  @ApiOperation({ summary: '获取按国家用户分布' })
  @ApiResponse({ status: 200, description: '按国家用户分布JSON列表' })
  async getStatsGeography() {
    return this.adminService.getStatsGeography();
  }

  @Get('stats/dau')
  @ApiOperation({ summary: '获取DAU趋势' })
  @ApiResponse({ status: 200, description: 'DAU趋势数据' })
  async getStatsDau(@Query('range') range: string = '7d'): Promise<{range: string, data: DauData[]}> {
    return this.adminService.getStatsDau(range);
  }
} 