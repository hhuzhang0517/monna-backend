import redis
import json
import sys
import argparse
from datetime import datetime

def format_json(data):
    """格式化JSON数据输出"""
    try:
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        if isinstance(data, str):
            return json.dumps(json.loads(data), indent=2, ensure_ascii=False)
        return json.dumps(data, indent=2, ensure_ascii=False)
    except:
        return str(data)

def check_redis_queue(host='localhost', port=6379, db=0, queue='ai_tasks'):
    """检查Redis队列状态和内容"""
    try:
        # 连接Redis
        r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        
        # 测试连接
        ping_result = r.ping()
        print(f"Redis连接测试: {'成功' if ping_result else '失败'}")
        
        # 获取队列长度
        queue_length = r.llen(queue)
        print(f"队列 '{queue}' 长度: {queue_length}")
        
        # 如果队列不为空，显示队列内容
        if queue_length > 0:
            print(f"\n队列 '{queue}' 内容:")
            # 不修改队列，仅查看内容
            queue_items = r.lrange(queue, 0, -1)
            for i, item in enumerate(queue_items):
                print(f"\n项目 {i+1}:")
                print(format_json(item))
        
        # 检查键空间
        print("\n所有相关的键:")
        for key in r.keys(f"*{queue}*"):
            print(f"  - {key}")
        
        # 检查任务状态键
        task_keys = r.keys("task:*")
        if task_keys:
            print("\n任务状态键:")
            for key in task_keys:
                data = r.hgetall(key)
                print(f"  - {key}: {format_json(data)}")
        
        # 检查发布/订阅通道
        print("\n发布/订阅通道:")
        pubsub = r.pubsub()
        pubsub.psubscribe("*")
        message = pubsub.get_message(timeout=0.01)
        channels = []
        while message:
            if message.get('type') == 'psubscribe':
                channels.append(message.get('channel'))
            message = pubsub.get_message(timeout=0.01)
        print(f"  活跃通道: {channels or '无'}")
        
        return True
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='检查Redis队列状态')
    parser.add_argument('--host', default='localhost', help='Redis主机')
    parser.add_argument('--port', type=int, default=6379, help='Redis端口')
    parser.add_argument('--db', type=int, default=0, help='Redis数据库索引')
    parser.add_argument('--queue', default='ai_tasks', help='要检查的队列名称')
    
    args = parser.parse_args()
    
    print(f"===== Redis队列检查 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) =====")
    print(f"连接到: {args.host}:{args.port}/{args.db}")
    print(f"队列名称: {args.queue}")
    print("="*50)
    
    success = check_redis_queue(args.host, args.port, args.db, args.queue)
    sys.exit(0 if success else 1) 