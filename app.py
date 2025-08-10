import streamlit as st
import os
import sys
import json
import time
import threading
from io import StringIO
import contextlib

# 设置页面配置
st.set_page_config(
    page_title="服装图像分类系统",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 添加自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .panel-header {
        font-size: 1.5rem;
        color: #333;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .terminal-output {
        background-color: #000;
        color: #00ff00;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        height: 300px;
        overflow-y: scroll;
    }
</style>
""", unsafe_allow_html=True)

# 主标题
st.markdown('<h1 class="main-header">👗 服装图像分类系统</h1>', unsafe_allow_html=True)

# 侧边栏导航
st.sidebar.title("系统面板")
page = st.sidebar.selectbox("选择功能模块", ["🏠 主页", "🎯 图像分类", "🔧 模型训练"])

# 初始化session state
if 'terminal_output' not in st.session_state:
    st.session_state.terminal_output = []
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def add_terminal_output(text):
    """添加输出到终端"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.terminal_output.append(f"[{timestamp}] {text}")
    if len(st.session_state.terminal_output) > 100:  # 限制输出行数
        st.session_state.terminal_output.pop(0)

def display_terminal():
    """显示终端输出"""
    st.subheader("📟 系统输出")
    terminal_content = "\n".join(st.session_state.terminal_output[-20:])  # 显示最近20行
    st.code(terminal_content, language="bash")
    
    # 清空终端按钮
    if st.button("🗑️ 清空输出"):
        st.session_state.terminal_output = []
        st.experimental_rerun()

# 主页面
if page == "🏠 主页":
    st.markdown('<h2 class="panel-header">系统概述</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🎯 图像分类功能**
        - 使用训练好的EfficientNetB0模型
        - 支持三类分类：主图、吊牌、细节
        - 批量处理图像
        - 自动移动分类结果
        """)
    
    with col2:
        st.info("""
        **🔧 模型训练功能**  
        - EfficientNetB0深度学习模型
        - 支持迁移学习和微调
        - 分阶段训练策略
        - 自动保存最佳模型
        """)
    
    st.markdown('<h2 class="panel-header">系统状态</h2>', unsafe_allow_html=True)
    
    # 检查模型文件
    model_paths = [
        "/home/runner/work/JiLing-baiditufenlei/JiLing-baiditufenlei/程序/模型/装载",
        "/home/runner/work/JiLing-baiditufenlei/JiLing-baiditufenlei/训练/新模型"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.endswith(('.keras', '.h5'))]
            if files:
                st.success(f"✅ 发现模型文件: {path} ({len(files)} 个文件)")
            else:
                st.warning(f"⚠️ 目录存在但无模型文件: {path}")
        else:
            st.error(f"❌ 模型目录不存在: {path}")
    
    display_terminal()

elif page == "🎯 图像分类":
    from 分类功能 import ClassificationModule
    classification_module = ClassificationModule()
    classification_module.render()
    
elif page == "🔧 模型训练":
    from 训练功能 import TrainingModule  
    training_module = TrainingModule()
    training_module.render()

# 底部信息
st.sidebar.markdown("---")
st.sidebar.info("🤖 基于EfficientNetB0深度学习模型\n📊 支持服装图像三分类")