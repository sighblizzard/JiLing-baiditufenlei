@echo off
echo 启动服装图像分类系统...
echo.
echo 正在检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到Python环境
    echo 请确保已安装Python 3.8或更高版本
    pause
    exit
)

echo.
echo 正在安装依赖包...
pip install -r requirements.txt

echo.
echo 启动Streamlit应用...
streamlit run app.py

pause