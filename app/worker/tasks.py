from celery import Task
import time
import logging
from pathlib import Path
from app.worker.celery_app import celery_app
from app.services.background import BackgroundRemovalService
from app.services.segmentation import BackgroundSegmentationService
from app.services.preprocessing import ImagePreprocessingService, ProcessingMode
from app.services.style_transfer import StyleTransferService
from app.utils.storage import StorageService
from app.services.facechain import FaceChainService

# 配置日志
logger = logging.getLogger(__name__)


class BaseTask(Task):
    """任务基类，提供通用功能和错误处理"""
    _storage = None
    
    @property
    def storage(self):
        if self._storage is None:
            self._storage = StorageService()
        return self._storage
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """处理任务失败"""
        logger.error(f"Celery Task {task_id} failed: {exc}", exc_info=True)
        # 更新任务状态，以便API可以查询到失败信息
        self.update_state(state='FAILURE', meta={'exc_type': type(exc).__name__, 'exc_message': str(exc)})
        return super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(bind=True, base=BaseTask)
def process_background_removal(self, image_path, options):
    """处理背景去除任务"""
    task_id = self.request.id
    logger.info(f"开始处理背景去除任务: {task_id}, 图片: {image_path}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '开始背景去除'
        })
        
        # 创建背景去除服务
        bg_service = BackgroundRemovalService()
        
        # 处理图像
        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '服务已加载，正在处理'
        })
        
        result = bg_service.remove_background(image_path, options)
        
        # 更新状态为50%
        self.update_state(state='PROCESSING', meta={
            'progress': 0.5,
            'message': '图像处理完成'
        })
        
        # 获取结果路径
        result_path = result.get('result_path')
        
        # 生成结果URL
        result_url = self.storage.get_file_url(result_path)
        
        logger.info(f"背景去除任务 {task_id} 完成, 结果URL: {result_url}")
        
        return {
            'status': 'COMPLETED',
            'result_url': result_url,
            'has_alpha': result.get('has_alpha', False),
            'progress': 1.0,
            'message': '背景去除成功完成'
        }
        
    except Exception as e:
        logger.error(f"背景去除任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_background_segmentation(self, image_path, options):
    """处理背景分割任务"""
    task_id = self.request.id
    logger.info(f"开始处理背景分割任务: {task_id}, 图片: {image_path}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '开始背景分割'
        })
        
        # 创建背景分割服务
        segmentation_service = BackgroundSegmentationService()
        
        # 处理图像
        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '服务已加载，正在执行图像分割'
        })
        
        result = segmentation_service.remove_background(image_path, options)
        
        # 更新状态为70%
        self.update_state(state='PROCESSING', meta={
            'progress': 0.7,
            'message': '分割完成，正在处理结果'
        })
        
        # 获取结果路径
        result_path = result.get('result_path')
        mask_path = result.get('mask_path')
        
        # 生成结果URL
        result_url = self.storage.get_file_url(result_path)
        mask_url = self.storage.get_file_url(mask_path) if mask_path else None
        
        logger.info(f"背景分割任务 {task_id} 完成, 结果URL: {result_url}")
        
        return {
            'status': 'COMPLETED',
            'result_url': result_url,
            'mask_url': mask_url,
            'has_alpha': result.get('has_alpha', False),
            'process_time': result.get('process_time', 0),
            'progress': 1.0,
            'message': '背景分割成功完成'
        }
        
    except Exception as e:
        logger.error(f"背景分割任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_cartoonization(self, image_path, options):
    """处理卡通化任务"""
    task_id = self.request.id
    logger.info(f"开始处理卡通化任务: {task_id}, 图片: {image_path}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '开始卡通化处理'
        })
        
        # 这里应该调用卡通化服务的实现
        # 暂时模拟处理
        time.sleep(2)
        
        # 模拟结果 - 在实际应用中，这里会调用真正的卡通化处理
        result_path = str(Path(image_path).with_suffix('.cartoon.jpg'))
        
        # 生成结果URL
        result_url = self.storage.get_file_url(result_path)
        
        # 任务完成
        self.update_state(state='PROCESSING', meta={
            'progress': 0.9,
            'message': '卡通化处理接近完成'
        })
        
        logger.info(f"卡通化任务 {task_id} 完成, 结果URL: {result_url}")
        
        return {
            'status': 'COMPLETED',
            'result_url': result_url,
            'progress': 1.0,
            'message': '卡通化处理成功完成'
        }
        
    except Exception as e:
        logger.error(f"卡通化任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_image_task(self, image_path, mode):
    """处理图像预处理任务"""
    task_id = self.request.id
    logger.info(f"开始处理图像预处理任务: {task_id}, 图片: {image_path}, 模式: {mode}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '开始图像预处理'
        })
        
        # 创建预处理服务
        preprocessor = ImagePreprocessingService()
        
        # 处理图像
        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '服务已加载，正在预处理'
        })
        
        # 执行预处理
        processing_mode = ProcessingMode(mode)
        result = preprocessor.process_image(image_path, mode=processing_mode)
        
        # 更新状态为进行中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.7,
            'message': '预处理完成'
        })
        
        # 任务完成
        self.update_state(state='PROCESSING', meta={
            'progress': 0.9,
            'message': '图像预处理成功完成'
        })
        
        # 返回预处理结果
        result['status'] = 'COMPLETED'
        result['progress'] = 1.0
        result['message'] = '图像预处理成功完成'
        return result
        
    except Exception as e:
        logger.error(f"图像预处理任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_style_transfer(self, image_paths, options):
    """处理风格转换任务"""
    task_id = self.request.id
    logger.info(f"开始处理风格转换任务: {task_id}, 图片数量: {len(image_paths)}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '开始风格转换'
        })
        
        # 获取风格类型
        style = options.get('style', 'wedding')
        logger.info(f"风格转换类型: {style}")
        
        # 创建风格转换服务
        style_service = StyleTransferService()
        
        # 更新状态
        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '服务已加载，正在生成风格图像'
        })
        
        # 执行风格转换
        result = style_service.transfer_style(image_paths, options)
        
        # 更新进度为80%
        self.update_state(state='PROCESSING', meta={
            'progress': 0.8,
            'message': '风格图像生成完成，准备结果'
        })
        
        # 准备返回结果
        storage = self.storage
        result_urls = [storage.get_file_url(path) for path in result.get('result_paths', [])]
        
        # 更新状态为完成
        self.update_state(state='PROCESSING', meta={
            'progress': 0.9,
            'message': '风格转换成功完成'
        })
        
        # 返回结果
        logger.info(f"风格转换任务 {task_id} 完成, 生成图像数量: {len(result_urls)}")
        return {
            'status': 'COMPLETED',
            'result_urls': result_urls,
            'style': style,
            'count': len(result_urls),
            'process_time': result.get('process_time', 0),
            'progress': 1.0,
            'message': '风格转换成功完成'
        }
        
    except Exception as e:
        logger.error(f"风格转换任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_facechain_portrait(self, task_data: dict):
    """处理FaceChain AI人像生成任务"""
    task_id = task_data.get('task_id', self.request.id)
    logger.info(f"开始处理FaceChain AI人像生成任务: Celery ID {self.request.id}, API Task ID {task_id}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '正在初始化FaceChain服务...'
        })
        
        facechain_service = FaceChainService()

        input_files = task_data.get('input_files', [])
        output_dir = task_data.get('output_dir')
        style = task_data.get('style')
        num_generate = task_data.get('num_generate', 1)
        multiplier_style = task_data.get('multiplier_style', 0.25)
        use_pose = task_data.get('use_pose', False)
        pose_file = task_data.get('pose_file')
        use_face_swap = task_data.get('use_face_swap', True)

        if not input_files:
            raise ValueError("没有提供输入图像文件")
        if not output_dir:
            raise ValueError("没有提供输出目录")
        if not style:
            raise ValueError("没有提供风格名称")

        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '参数校验完成，正在生成AI人像...'
        })
        primary_input = input_files[0]

        result_files = facechain_service.generate_portrait(
            input_img_path=primary_input,
            output_dir=output_dir,
            style_name=style,
            num_generate=num_generate,
            multiplier_style=multiplier_style,
            use_pose=use_pose,
            pose_image_path=pose_file,
            use_face_swap=use_face_swap
        )
        self.update_state(state='PROCESSING', meta={
            'progress': 0.9,
            'message': 'AI人像生成完成，处理结果...'
        })

        result_urls = [self.storage.get_file_url(path) for path in result_files if path]

        logger.info(f"FaceChain AI人像生成任务 {task_id} 完成, 生成图像数量: {len(result_urls)}")
        return {
            'status': 'COMPLETED',
            'result_urls': result_urls,
            'style': style,
            'count': len(result_urls),
            'progress': 1.0,
            'message': f'成功生成 {len(result_urls)} 张AI人像'
        }
    except Exception as e:
        logger.error(f"FaceChain AI人像生成任务 {task_id} 失败: {e}", exc_info=True)
        raise


@celery_app.task(bind=True, base=BaseTask)
def process_facechain_inpaint(self, task_data: dict):
    """处理FaceChain AI人像修复任务"""
    task_id = task_data.get('task_id', self.request.id)
    logger.info(f"开始处理FaceChain AI人像修复任务: Celery ID {self.request.id}, API Task ID {task_id}")
    
    try:
        # 更新任务状态为处理中
        self.update_state(state='PROCESSING', meta={
            'progress': 0.1,
            'message': '正在初始化FaceChain修复服务...'
        })
        
        facechain_service = FaceChainService()

        input_files = task_data.get('input_files', [])
        template_file = task_data.get('template_file')
        output_dir = task_data.get('output_dir')
        num_faces = task_data.get('num_faces', 1)
        selected_face = task_data.get('selected_face', 1)
        use_face_swap = task_data.get('use_face_swap', True)

        if not input_files:
            raise ValueError("没有提供输入图像文件")
        if not template_file:
            raise ValueError("没有提供模板图像文件")
        if not output_dir:
            raise ValueError("没有提供输出目录")

        self.update_state(state='PROCESSING', meta={
            'progress': 0.3,
            'message': '参数校验完成，正在生成修复版AI人像...'
        })
        primary_input = input_files[0]

        result_files = facechain_service.generate_portrait_inpaint(
            input_img_path=primary_input,
            template_img_path=template_file,
            output_dir=output_dir,
            num_faces=num_faces,
            selected_face=selected_face,
            use_face_swap=use_face_swap
        )
        self.update_state(state='PROCESSING', meta={
            'progress': 0.9,
            'message': '修复版AI人像生成完成，处理结果...'
        })

        result_urls = [self.storage.get_file_url(path) for path in result_files if path]
        
        logger.info(f"FaceChain AI人像修复任务 {task_id} 完成, 生成图像数量: {len(result_urls)}")
        return {
            'status': 'COMPLETED',
            'result_urls': result_urls,
            'count': len(result_urls),
            'progress': 1.0,
            'message': f'成功生成 {len(result_urls)} 张修复版AI人像'
        }
    except Exception as e:
        logger.error(f"FaceChain AI人像修复任务 {task_id} 失败: {e}", exc_info=True)
        raise