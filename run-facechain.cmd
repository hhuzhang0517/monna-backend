@echo off
echo 正在启动Monna AI FaceChain后端服务...

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 设置环境变量
set PYTHONIOENCODING=utf-8

:: 检查Python是否可用
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python解释器，请确保Python已安装并添加到PATH环境变量中
    pause
    exit /b 1
)

:: 启动服务
echo 启动FastAPI服务...
python main.py

:: 如果服务退出
echo 服务已退出
pause 