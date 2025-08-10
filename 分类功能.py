import streamlit as st
import os
import sys
import shutil
import numpy as np
import json
import time
import threading
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow.keras.models
import tensorflow.keras.preprocessing.image

class ClassificationModule:
    def __init__(self):
        self.base_path = os.path.abspath(os.path.dirname(__file__))
        
    def add_terminal_output(self, text):
        """添加输出到终端"""
        timestamp = time.strftime("%H:%M:%S")
        if 'terminal_output' not in st.session_state:
            st.session_state.terminal_output = []
        st.session_state.terminal_output.append(f"[{timestamp}] {text}")
        if len(st.session_state.terminal_output) > 100:
            st.session_state.terminal_output.pop(0)
    
    def find_model_file(self, model_dir):
        """查找模型文件"""
        if not os.path.exists(model_dir):
            return None
        files = [f for f in os.listdir(model_dir) if f.lower().endswith(('.keras', '.h5'))]
        if not files:
            return None
        files = sorted(
            files,
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        return os.path.join(model_dir, files[0])
    
    def load_class_names(self, model_dir):
        """加载类别名称"""
        class_names_file = os.path.join(model_dir, 'class_names.json')
        if os.path.exists(class_names_file):
            try:
                with open(class_names_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.add_terminal_output(f"加载类别文件失败: {e}")
                return ['主图', '吊牌', '细节']
        return ['主图', '吊牌', '细节']
    
    def classify_images(self, model_path, input_dir, output_base, class_names, 
                       img_height, img_width, batch_size, confidence_threshold, 
                       progress_bar, status_text):
        """图像分类主函数"""
        try:
            # 加载模型
            status_text.text("正在加载模型...")
            model = load_model(model_path)
            self.add_terminal_output(f"模型加载成功: {model_path}")
            
            # 验证模型输出维度
            expected_classes = len(class_names)
            actual_classes = model.output_shape[-1]
            if actual_classes != expected_classes:
                self.add_terminal_output(f"警告：模型输出维度({actual_classes})与类别数({expected_classes})不匹配！")
                if actual_classes < expected_classes:
                    class_names = class_names[:actual_classes]
                    self.add_terminal_output(f"已调整类别列表为: {class_names}")
            
            # 检查输入目录
            if not os.path.exists(input_dir):
                self.add_terminal_output(f"输入目录不存在: {input_dir}")
                return False
            
            # 获取所有图片文件
            img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
            files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
            total = len(files)
            if total == 0:
                self.add_terminal_output("没有检测到图片文件")
                return False
            
            self.add_terminal_output(f"共检测到 {total} 张图片，开始批量分类...")
            
            # 创建输出目录
            output_dirs = {}
            for class_name in class_names:
                output_dirs[class_name] = os.path.join(output_base, class_name)
                os.makedirs(output_dirs[class_name], exist_ok=True)
            
            # 批量分类
            processed = 0
            successful = 0
            failed = 0
            low_confidence_count = 0
            category_counts = {name: 0 for name in class_names}
            
            status_text.text("正在处理图像...")
            
            for i in range(0, total, batch_size):
                batch_files = files[i:i+batch_size]
                batch_imgs = []
                valid_files = []
                
                # 进度更新
                progress = min((i + batch_size) / total, 1.0)
                progress_bar.progress(progress)
                
                self.add_terminal_output(f"处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch_files)} 张图片)")
                
                for file in batch_files:
                    img_path = os.path.join(input_dir, file)
                    try:
                        img = image.load_img(img_path, target_size=(img_height, img_width))
                        img_array = image.img_to_array(img) / 255.0
                        batch_imgs.append(img_array)
                        valid_files.append(file)
                    except Exception as e:
                        self.add_terminal_output(f"加载图片 {file} 时出错：{e}")
                        failed += 1
                
                if not batch_imgs:
                    continue
                
                try:
                    batch_imgs_np = np.array(batch_imgs)
                    preds = model.predict(batch_imgs_np, verbose=0)
                    
                    for j, pred in enumerate(preds):
                        file = valid_files[j]
                        img_path = os.path.join(input_dir, file)
                        processed += 1
                        
                        pred_class = np.argmax(pred)
                        max_confidence = np.max(pred)
                        pred_label = class_names[pred_class]
                        
                        # 检查置信度
                        if max_confidence < confidence_threshold:
                            # 低置信度图片移动到细节文件夹
                            detail_dir = output_dirs.get('细节', os.path.join(output_base, '细节'))
                            os.makedirs(detail_dir, exist_ok=True)
                            shutil.copy(img_path, os.path.join(detail_dir, file))
                            self.add_terminal_output(f"{file} → 细节(低置信度:{max_confidence:.3f}) → 原预测:{pred_label}")
                            low_confidence_count += 1
                            category_counts['细节'] += 1
                        else:
                            if pred_label in output_dirs:
                                out_dir = output_dirs[pred_label]
                                shutil.copy(img_path, os.path.join(out_dir, file))
                                self.add_terminal_output(f"{file} → {pred_label} (置信度: {max_confidence:.3f})")
                                successful += 1
                                category_counts[pred_label] += 1
                            else:
                                self.add_terminal_output(f"{file} → 预测为 {pred_label}，不做移动")
                                failed += 1
                
                except Exception as e:
                    self.add_terminal_output(f"批次预测失败: {e}")
                    failed += len(valid_files)
            
            progress_bar.progress(1.0)
            status_text.text("分类完成！")
            
            # 输出统计结果
            self.add_terminal_output("=== 分类完成 ===")
            self.add_terminal_output(f"总处理: {processed} 张")
            self.add_terminal_output(f"成功分类: {successful} 张")
            self.add_terminal_output(f"低置信度(归入细节): {low_confidence_count} 张")
            self.add_terminal_output(f"失败: {failed} 张")
            
            self.add_terminal_output("=== 各类别统计 ===")
            for class_name, count in category_counts.items():
                self.add_terminal_output(f"{class_name}: {count} 张")
            
            return True
            
        except Exception as e:
            self.add_terminal_output(f"分类过程发生错误：{e}")
            import traceback
            self.add_terminal_output(traceback.format_exc())
            return False
    
    def render(self):
        """渲染分类面板"""
        st.markdown('<h2 class="panel-header">🎯 图像分类面板</h2>', unsafe_allow_html=True)
        
        # 参数设置区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 分类参数设置")
            
            # 模型设置
            model_dir = st.text_input(
                "模型目录路径",
                value=os.path.join(self.base_path, "程序", "模型", "装载"),
                help="包含模型文件(.keras或.h5)的目录路径"
            )
            
            # 输入输出路径设置  
            input_dir = st.text_input(
                "输入图片目录",
                value=r"D:\桌面\筛选\JPG",
                help="要分类的图片所在目录"
            )
            
            output_base = st.text_input(
                "输出基础目录", 
                value=r"D:\桌面\筛选",
                help="分类结果存放的基础目录，会在此目录下创建子目录"
            )
            
            # 模型参数设置
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                img_size = st.selectbox("图像尺寸", [512, 448, 384, 256], index=0)
                batch_size = st.slider("批处理大小", min_value=8, max_value=64, value=48, step=8)
            
            with col_param2:
                confidence_threshold = st.slider("置信度阈值", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
                delete_original = st.checkbox("处理后删除原始文件", value=False)
        
        with col2:
            st.subheader("📊 模型信息")
            
            # 检查模型状态
            if os.path.exists(model_dir):
                model_path = self.find_model_file(model_dir)
                if model_path:
                    st.success("✅ 模型文件存在")
                    st.info(f"模型文件: {os.path.basename(model_path)}")
                    
                    # 加载类别信息
                    class_names = self.load_class_names(model_dir)
                    st.info(f"分类类别: {', '.join(class_names)}")
                else:
                    st.error("❌ 未找到模型文件")
            else:
                st.error("❌ 模型目录不存在")
            
            # 检查输入目录
            if os.path.exists(input_dir):
                img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
                files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
                st.info(f"待处理图片: {len(files)} 张")
            else:
                st.warning("⚠️ 输入目录不存在")
        
        # 分类操作区域
        st.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            start_classification = st.button("🚀 开始分类", type="primary", disabled=st.session_state.is_running)
        
        with col_btn2:
            if st.button("🔄 刷新状态"):
                st.experimental_rerun()
        
        # 执行分类
        if start_classification:
            if not os.path.exists(model_dir):
                st.error("❌ 模型目录不存在，请检查路径")
                return
            
            model_path = self.find_model_file(model_dir)
            if not model_path:
                st.error("❌ 未找到模型文件")
                return
            
            if not os.path.exists(input_dir):
                st.error("❌ 输入目录不存在，请检查路径")
                return
            
            class_names = self.load_class_names(model_dir)
            
            # 创建进度条和状态显示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            st.session_state.is_running = True
            
            # 执行分类
            success = self.classify_images(
                model_path, input_dir, output_base, class_names,
                img_size, img_size, batch_size, confidence_threshold,
                progress_bar, status_text
            )
            
            st.session_state.is_running = False
            
            if success:
                st.success("🎉 图像分类完成！")
                if delete_original:
                    # 删除原始文件
                    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
                    files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
                    deleted_count = 0
                    for file in files:
                        try:
                            os.remove(os.path.join(input_dir, file))
                            deleted_count += 1
                        except Exception as e:
                            self.add_terminal_output(f"删除 {file} 时出错：{e}")
                    self.add_terminal_output(f"已删除 {deleted_count} 个原始文件")
            else:
                st.error("❌ 分类过程中发生错误，请查看终端输出")
        
        # 显示终端输出
        st.markdown("---")
        st.subheader("📟 系统输出")
        if 'terminal_output' in st.session_state and st.session_state.terminal_output:
            terminal_content = "\n".join(st.session_state.terminal_output[-20:])
            st.code(terminal_content, language="bash")
        else:
            st.info("暂无输出信息")
        
        # 清空终端按钮
        if st.button("🗑️ 清空输出"):
            st.session_state.terminal_output = []
            st.experimental_rerun()