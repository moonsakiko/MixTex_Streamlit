# 文件名: streamlit_app.py

import streamlit as st
from PIL import Image
# ⬇️⬇️⬇️ 引入我们需要的其他模块 ⬇️⬇️⬇️
from processor import load_model, pad_image, stream_inference 
import re # 引入正则表达式模块

# --- 页面基础配置 ---
st.set_page_config(page_title="MixTeX LaTeX OCR 在线工具", page_icon="📝")
st.title("📝 MixTeX - LaTeX/公式/表格 OCR")
st.info("上传一张包含公式、表格或中英文文本的图片，AI将自动识别并返回LaTeX或纯文本结果。")
st.warning("模型较大，首次加载可能需要1-2分钟，请耐心等待。")

# ===================================================================
# ---                         核心逻辑部分                          ---
# ===================================================================

# 加载模型 (只会执行一次)
model = load_model()

if model:
    # --- UI 界面 ---
    uploaded_file = st.file_uploader("上传图片文件 (支持PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

    # ⬇️⬇️⬇️ 新增：输出格式选择器 ⬇️⬇️⬇️
    output_format = st.radio(
        "选择输出格式",
        ("兼容模式 (推荐)", "专业模式 (原始输出)"),
        horizontal=True,
        help="""
        - **兼容模式**: 为多行公式自动添加 `\\begin{align*}` 环境，能被Obsidian和大多数在线编辑器正确渲染。\n
        - **专业模式**: 提供模型最原始的输出，适合需要手动编辑的LaTeX专家。
        """
    )
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("图片预览")
                st.image(img, caption="上传图片")
            
            with col2:
                st.subheader("识别结果")
                if st.button("开始识别", use_container_width=True):
                    with st.spinner("AI 正在识别中..."):
                        # 预处理图片
                        img_padded = pad_image(img)
                        
                        # 流式获取原始结果
                        raw_result = ""
                        for piece in stream_inference(img_padded, model):
                            raw_result += piece
                        
                        # --- ⬇️⬇️⬇️ 根据用户选择，对结果进行后处理 ⬇️⬇️⬇️ ---
                        final_result = raw_result
                        
                        if output_format == "兼容模式 (推荐)":
                            # 使用正则表达式，智能地为包含 align 标记的公式块添加环境
                            # 查找以 \[ 开头，以 \] 结尾，并且中间包含 & 或 \\ 的块
                            def add_align_wrapper(match):
                                content = match.group(1)
                                # 如果内容里真的有对齐或换行符，才添加包裹
                                if '&' in content or '\\\\' in content:
                                    return f"\\begin{{align*}}\n{content}\n\\end{{align*}}"
                                else:
                                    # 否则保持原样
                                    return f"\\[\n{content}\n\\]"
                            
                            # re.DOTALL 让 . 可以匹配换行符
                            final_result = re.sub(r"\\\[\s*(.*?)\s*\\\]", add_align_wrapper, raw_result, flags=re.DOTALL)

                        # --- 最终显示和复制 ---
                        st.markdown(f"```latex\n{final_result}\n```")
                        st.success("识别完成！点击下方按钮可复制结果。")
                        st.code(final_result, language="latex")

        except Exception as e:
            st.error(f"处理图片时出错: {e}")
else:
    st.error("模型未能加载，应用无法运行。请检查日志。")