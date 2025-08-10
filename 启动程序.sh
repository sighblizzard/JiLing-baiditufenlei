#!/bin/bash
echo "启动服装图像分类系统..."
echo

echo "正在检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python环境"
    echo "请确保已安装Python 3.8或更高版本"
    read -p "按回车键退出..."
    exit 1
fi

echo
echo "正在安装依赖包..."
pip3 install -r requirements.txt

echo
echo "启动Streamlit应用..."
streamlit run app.py

echo "按回车键退出..."
read