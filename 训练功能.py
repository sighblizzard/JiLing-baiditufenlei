import streamlit as st
import os
import sys
import json
import time
import threading
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import pandas as pd

class TrainingModule:
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
        files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
        return os.path.join(model_dir, files[0])
    
    def train_model(self, base_dir, model_load_dir, model_save_dir, img_height, img_width, 
                   batch_size, epochs_stage1, epochs_stage2, epochs_stage3, 
                   learning_rate, validation_split, progress_container, status_text):
        """模型训练主函数"""
        try:
            # 数据增强设置
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.15,
                height_shift_range=0.15,
                zoom_range=0.15,
                horizontal_flip=True,
                brightness_range=[0.8, 1.2],
                shear_range=0.1,
                fill_mode='nearest',
                validation_split=validation_split
            )
            
            val_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split
            )
            
            # 创建数据生成器
            train_generator = train_datagen.flow_from_directory(
                base_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode="categorical",
                subset="training",
                shuffle=True,
                seed=42
            )
            
            val_generator = val_datagen.flow_from_directory(
                base_dir,
                target_size=(img_height, img_width),
                batch_size=batch_size,
                class_mode="categorical",
                subset="validation",
                shuffle=False,
                seed=42
            )
            
            # 获取类别信息
            class_indices = train_generator.class_indices
            class_names = [None] * len(class_indices)
            for name, idx in class_indices.items():
                class_names[idx] = name
            num_classes = len(class_names)
            
            self.add_terminal_output(f"检测到类别 ({num_classes}): {class_names}")
            self.add_terminal_output(f"训练集样本数: {train_generator.samples}")
            self.add_terminal_output(f"验证集样本数: {val_generator.samples}")
            
            # 保存类别信息
            os.makedirs(model_save_dir, exist_ok=True)
            with open(os.path.join(model_save_dir, "class_names.json"), "w", encoding="utf-8") as jf:
                json.dump(class_names, jf, ensure_ascii=False)
            
            # 尝试加载已有模型
            model_path = self.find_model_file(model_load_dir) if model_load_dir else None
            base_model = None
            
            if model_path and os.path.exists(model_path):
                self.add_terminal_output(f"尝试加载已有模型: {model_path}")
                try:
                    model = load_model(model_path)
                    self.add_terminal_output(f"成功加载模型，输出类别数: {model.output_shape[-1]}")
                    
                    # 验证模型输出维度
                    if model.output_shape[-1] != num_classes:
                        self.add_terminal_output(f"警告：模型输出维度({model.output_shape[-1]})与数据集类别数({num_classes})不匹配！")
                        self.add_terminal_output("将重新构建模型...")
                        model_path = None
                    else:
                        # 尝试获取base_model
                        for layer in model.layers:
                            if hasattr(layer, 'name') and 'efficientnet' in layer.name.lower():
                                base_model = layer
                                break
                except Exception as e:
                    self.add_terminal_output(f"加载模型失败: {e}")
                    model_path = None
            
            # 构建新模型
            if not model_path:
                self.add_terminal_output("构建新的EfficientNetB0模型...")
                base_model = EfficientNetB0(
                    include_top=False, weights='imagenet',
                    input_shape=(img_height, img_width, 3)
                )
                inputs = keras.Input(shape=(img_height, img_width, 3))
                x = base_model(inputs, training=False)
                x = keras.layers.GlobalAveragePooling2D()(x)
                x = keras.layers.Dropout(0.3)(x)
                x = keras.layers.Dense(256, activation='relu')(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(0.2)(x)
                outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
                model = keras.Model(inputs, outputs)
                self.add_terminal_output(f"新模型构建完成，类别数: {num_classes}")
            
            # 显示模型信息
            total_params = model.count_params()
            trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
            self.add_terminal_output(f"总参数数: {total_params:,}")
            self.add_terminal_output(f"可训练参数数: {trainable_params:,}")
            
            # 设置回调函数
            checkpoint = callbacks.ModelCheckpoint(
                os.path.join(model_save_dir, "best_model.keras"), 
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
                verbose=1
            )
            
            early_stop = callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=8,
                restore_best_weights=True,
                mode='max',
                verbose=1
            )
            
            lr_scheduler = callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
            
            # 自定义进度回调
            class StreamlitProgressCallback(callbacks.Callback):
                def __init__(self, progress_container, status_text, total_epochs):
                    super().__init__()
                    self.progress_container = progress_container
                    self.status_text = status_text
                    self.total_epochs = total_epochs
                    self.current_epoch = 0
                    self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
                
                def on_epoch_begin(self, epoch, logs=None):
                    self.current_epoch = epoch
                    self.status_text.text(f"训练第 {epoch + 1}/{self.total_epochs} 轮...")
                
                def on_epoch_end(self, epoch, logs=None):
                    # 更新历史记录
                    for key in self.history.keys():
                        if key in logs:
                            self.history[key].append(logs[key])
                    
                    # 更新进度条
                    progress = (epoch + 1) / self.total_epochs
                    
                    with self.progress_container:
                        st.progress(progress)
                        
                        # 显示训练指标
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("训练损失", f"{logs.get('loss', 0):.4f}")
                        with col2:
                            st.metric("训练准确率", f"{logs.get('accuracy', 0):.4f}")
                        with col3:
                            st.metric("验证损失", f"{logs.get('val_loss', 0):.4f}")
                        with col4:
                            st.metric("验证准确率", f"{logs.get('val_accuracy', 0):.4f}")
                        
                        # 显示训练曲线
                        if len(self.history['loss']) > 1:
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # 损失曲线
                            ax1.plot(self.history['loss'], label='训练损失')
                            ax1.plot(self.history['val_loss'], label='验证损失')
                            ax1.set_title('模型损失')
                            ax1.set_xlabel('轮次')
                            ax1.set_ylabel('损失')
                            ax1.legend()
                            
                            # 准确率曲线
                            ax2.plot(self.history['accuracy'], label='训练准确率')
                            ax2.plot(self.history['val_accuracy'], label='验证准确率')
                            ax2.set_title('模型准确率')
                            ax2.set_xlabel('轮次')
                            ax2.set_ylabel('准确率')
                            ax2.legend()
                            
                            st.pyplot(fig)
                            plt.close(fig)
            
            # 分阶段训练
            total_epochs = epochs_stage1 + epochs_stage2 + epochs_stage3
            progress_callback = StreamlitProgressCallback(progress_container, status_text, total_epochs)
            
            # 阶段1：冻结基础模型
            if base_model:
                base_model.trainable = False
            
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
            
            self.add_terminal_output("阶段1：训练分类头...")
            history1 = model.fit(
                train_generator,
                epochs=epochs_stage1,
                validation_data=val_generator,
                callbacks=[progress_callback],
                verbose=0
            )
            
            # 阶段2：解冻部分层
            if base_model and epochs_stage2 > 0:
                base_model.trainable = True
                for layer in base_model.layers[:-30]:
                    layer.trainable = False
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate/10),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )
                
                self.add_terminal_output("阶段2：部分精调...")
                progress_callback.total_epochs = epochs_stage2
                progress_callback.current_epoch = 0
                history2 = model.fit(
                    train_generator,
                    epochs=epochs_stage2,
                    validation_data=val_generator,
                    callbacks=[early_stop, checkpoint, progress_callback],
                    verbose=0
                )
            
            # 阶段3：全网络精调
            if base_model and epochs_stage3 > 0:
                for layer in base_model.layers:
                    layer.trainable = True
                
                model.compile(
                    optimizer=keras.optimizers.Adam(learning_rate/100),
                    loss="categorical_crossentropy",
                    metrics=["accuracy"]
                )
                
                self.add_terminal_output("阶段3：全网络精调...")
                progress_callback.total_epochs = epochs_stage3
                progress_callback.current_epoch = 0
                history3 = model.fit(
                    train_generator,
                    epochs=epochs_stage3,
                    validation_data=val_generator,
                    callbacks=[early_stop, checkpoint, lr_scheduler, progress_callback],
                    verbose=0
                )
            
            # 保存最终模型
            final_model_path = os.path.join(model_save_dir, "final_model.keras")
            model.save(final_model_path)
            self.add_terminal_output(f"最终模型已保存到: {final_model_path}")
            
            return True
            
        except Exception as e:
            self.add_terminal_output(f"训练过程发生错误：{e}")
            import traceback
            self.add_terminal_output(traceback.format_exc())
            return False
    
    def render(self):
        """渲染训练面板"""
        st.markdown('<h2 class="panel-header">🔧 模型训练面板</h2>', unsafe_allow_html=True)
        
        # 参数设置区域
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 训练参数设置")
            
            # 数据路径设置
            base_dir = st.text_input(
                "训练数据目录",
                value=os.path.join(self.base_path, "筛选"),
                help="包含各类别子目录的训练数据根目录"
            )
            
            model_load_dir = st.text_input(
                "预训练模型目录（可选）",
                value=os.path.join(self.base_path, "训练", "模型装载"),
                help="用于继续训练的预训练模型目录"
            )
            
            model_save_dir = st.text_input(
                "模型保存目录",
                value=os.path.join(self.base_path, "训练", "新模型"),
                help="训练完成的模型保存目录"
            )
            
            # 训练参数设置
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                img_size = st.selectbox("图像尺寸", [512, 448, 384, 256], index=0)
                batch_size = st.slider("批处理大小", min_value=8, max_value=64, value=16, step=8)
                learning_rate = st.selectbox("学习率", [0.001, 0.0001, 0.00001], index=1)
            
            with col_param2:
                validation_split = st.slider("验证集比例", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
                epochs_stage1 = st.number_input("阶段1训练轮数", min_value=5, max_value=50, value=10)
                epochs_stage2 = st.number_input("阶段2训练轮数", min_value=0, max_value=50, value=10)
            
            epochs_stage3 = st.number_input("阶段3训练轮数", min_value=0, max_value=50, value=15)
        
        with col2:
            st.subheader("📊 数据集信息")
            
            # 检查数据集状态
            if os.path.exists(base_dir):
                categories = [d for d in os.listdir(base_dir) 
                             if os.path.isdir(os.path.join(base_dir, d))]
                if categories:
                    st.success(f"✅ 发现 {len(categories)} 个类别")
                    for category in categories:
                        category_path = os.path.join(base_dir, category)
                        img_files = [f for f in os.listdir(category_path) 
                                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        st.info(f"{category}: {len(img_files)} 张图片")
                else:
                    st.warning("⚠️ 未发现类别目录")
            else:
                st.error("❌ 训练数据目录不存在")
            
            # 检查预训练模型
            if model_load_dir and os.path.exists(model_load_dir):
                model_path = self.find_model_file(model_load_dir)
                if model_path:
                    st.success("✅ 发现预训练模型")
                    st.info(f"模型: {os.path.basename(model_path)}")
                else:
                    st.warning("⚠️ 未发现预训练模型")
            else:
                st.info("ℹ️ 将从头开始训练")
        
        # 训练操作区域
        st.markdown("---")
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            start_training = st.button("🚀 开始训练", type="primary", disabled=st.session_state.is_running)
        
        with col_btn2:
            if st.button("🔄 刷新状态"):
                st.experimental_rerun()
        
        # 执行训练
        if start_training:
            if not os.path.exists(base_dir):
                st.error("❌ 训练数据目录不存在，请检查路径")
                return
            
            categories = [d for d in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, d))]
            if len(categories) < 2:
                st.error("❌ 至少需要2个类别目录")
                return
            
            # 检查每个类别是否有足够的图片
            min_images = 5
            for category in categories:
                category_path = os.path.join(base_dir, category)
                img_files = [f for f in os.listdir(category_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if len(img_files) < min_images:
                    st.error(f"❌ 类别 '{category}' 图片数量不足 (至少需要{min_images}张)")
                    return
            
            st.info("🚀 开始模型训练...")
            st.session_state.is_running = True
            
            # 创建进度显示容器
            progress_container = st.container()
            status_text = st.empty()
            
            # 执行训练
            success = self.train_model(
                base_dir, model_load_dir, model_save_dir, 
                img_size, img_size, batch_size,
                epochs_stage1, epochs_stage2, epochs_stage3,
                learning_rate, validation_split,
                progress_container, status_text
            )
            
            st.session_state.is_running = False
            
            if success:
                st.success("🎉 模型训练完成！")
                st.balloons()
            else:
                st.error("❌ 训练过程中发生错误，请查看终端输出")
        
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