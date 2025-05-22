@echo off
echo 正在启动Monna AI FaceChain后端服务...

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查虚拟环境是否存在
if not exist "venv310\Scripts\activate.bat" (
    echo 错误: 未找到Python虚拟环境，请确保已安装并配置Python 3.10虚拟环境
    pause
    exit /b 1
)

:: 激活虚拟环境
echo 激活Python虚拟环境...
call venv310\Scripts\activate.bat

:: 设置环境变量
set PYTHONIOENCODING=utf-8

:: 启动服务
echo 启动FastAPI服务...
python main.py

:: 如果服务退出
echo 服务已退出
pause 