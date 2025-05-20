import os
import shutil
from pathlib import Path
from typing import BinaryIO, Optional
import aiofiles
import uuid
from fastapi import UploadFile
from app.core.config import settings

class StorageService:
    @staticmethod
    async def save_upload(upload_file: UploadFile, subdir: Optional[str] = None) -> str:
        """异步保存上传文件"""
        # 确定保存目录
        save_dir = settings.UPLOAD_DIR
        if subdir:
            save_dir = save_dir / subdir
            os.makedirs(save_dir, exist_ok=True)
            
        # 生成唯一文件名
        ext = os.path.splitext(upload_file.filename)[1]
        filename = f"{uuid.uuid4()}{ext}"
        file_path = save_dir / filename
        
        # 异步保存文件
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await upload_file.read()
            await out_file.write(content)
            
        return str(file_path)
    
    @staticmethod
    def save_result(file_data: bytes, extension: str = ".jpg", subdir: Optional[str] = None) -> str:
        """保存处理结果文件"""
        # 确定保存目录
        save_dir = settings.RESULTS_DIR
        if subdir:
            save_dir = save_dir / subdir
            os.makedirs(save_dir, exist_ok=True)
            
        # 生成唯一文件名
        filename = f"{uuid.uuid4()}{extension}"
        file_path = save_dir / filename
        
        # 保存文件
        with open(file_path, 'wb') as out_file:
            out_file.write(file_data)
            
        return str(file_path)
    
    @staticmethod
    def get_file_url(file_path: str) -> str:
        """根据文件路径生成URL"""
        # 这里简化处理，实际生产环境可能需要考虑CDN等因素
        relative_path = Path(file_path).relative_to(settings.BASE_DIR)
        return f"/static/{relative_path}"
    
    @staticmethod
    def cleanup_old_files(directory: Path, max_age_days: int = 1):
        """清理指定天数前的文件"""
        import time
        
        now = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for file_path in directory.glob('**/*'):
            if file_path.is_file():
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    os.remove(file_path)