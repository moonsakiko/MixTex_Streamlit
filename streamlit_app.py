# 文件名: streamlit_app.py

import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoImageProcessor
import re
import io

# --- 页面基础配置 ---
st.set_page_config(page_title="MixTeX LaTeX OCR 在线工具", page_icon="📝")
st.title("📝 MixTeX - LaTeX/公式/表格 OCR")
st.info("上传一张包含公式、表格或中英文文本的图片，AI将自动识别并返回LaTeX或纯文本结果。")
st.warning("模型较大，首次加载可能需要1-2分钟，请耐心等待。")

# ===================================================================
# ---                         核心逻辑部分 (心脏移植)                 ---
# ===================================================================

@st.cache_resource
def load_model(model_dir="onnx"):
    """
    加载所有必要的模型和分词器。
    模型文件需要放在一个名为 'onnx' 的文件夹里。
    """
    try:
        # 确保Hugging Face的缓存目录存在
        # from pathlib import Path
        # Path("/app/.cache/huggingface/hub").mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        feature_extractor = AutoImageProcessor.from_pretrained(model_dir)
        
        # 为ONNX模型指定providers，优先使用CPU
        providers = ['CPUExecutionProvider']
        encoder_sess = ort.InferenceSession(f"{model_dir}/encoder_model.onnx", providers=providers)
        decoder_sess = ort.InferenceSession(f"{model_dir}/decoder_model_merged.onnx", providers=providers)
        
        return tokenizer, feature_extractor, encoder_sess, decoder_sess
    except Exception as e:
        # 提供更详细的错误，方便排错
        st.error(f"模型加载失败！请确保'onnx'文件夹包含所有模型文件，并且位于项目根目录。错误: {e}")
        return None

def pad_image(img, out_size=(448, 448)):
    """将图片缩放并填充到目标尺寸"""
    x_img, y_img = out_size
    bg = Image.new("RGB", (x_img, y_img), (255, 255, 255))
    w, h = img.size
    if w > x_img or h > y_img:
        scale = min(x_img / w, y_img / h)
        nw, nh = int(w * scale), int(h * scale)
        img = img.resize((nw, nh), Image.LANCZOS)
    
    w, h = img.size
    x = (x_img - w) // 2
    y = (y_img - h) // 2
    bg.paste(img, (x, y))
    return bg

def check_repetition(s, repeats=12):
    """检查生成文本中是否有过度重复"""
    if len(s) < repeats: return False
    for pattern_length in range(1, len(s) // repeats + 1):
        pattern = s[-pattern_length:]
        if s[-repeats * pattern_length:] == pattern * repeats:
            return True
    return False

def stream_inference(image, model, max_length=512, num_layers=6, hidden_size=768, heads=12, batch_size=1):
    """流式推理函数，逐个token生成结果"""
    tokenizer, feature_extractor, enc_session, dec_session = model
    head_size = hidden_size // heads
    
    inputs = feature_extractor(image, return_tensors="np").pixel_values
    enc_out = enc_session.run(None, {"pixel_values": inputs})[0]
    
    dec_in = {
        "input_ids": tokenizer("<s>", return_tensors="np").input_ids.astype(np.int64),
        "encoder_hidden_states": enc_out,
        "use_cache_branch": np.array([True], dtype=bool),
        **{
            f"past_key_values.{i}.{t}": np.zeros(
                (batch_size, heads, 0, head_size), dtype=np.float32
            )
            for i in range(num_layers)
            for t in ["key", "value"]
        },
    }
    
    generated = ""
    for _ in range(max_length):
        outs = dec_session.run(None, dec_in)
        next_id = np.argmax(outs[0][:, -1, :], axis=-1)
        
        # 检查是否是结束符
        if next_id[0] == tokenizer.eos_token_id:
            break
            
        token_text = tokenizer.decode(next_id, skip_special_tokens=True)
        yield token_text
        
        generated += token_text
        if check_repetition(generated, 21):
            break
            
        dec_in.update(
            {
                "input_ids": next_id[:, None],
                **{
                    f"past_key_values.{i}.{t}": outs[i * 2 + 1 + j]
                    for i in range(num_layers)
                    for j, t in enumerate(["key", "value"])
                },
            }
        )

# ===================================================================
# ---                         前端界面部分 (新车身)                   ---
# ===================================================================

# 加载模型 (只会执行一次)
model = load_model()

if model:
    uploaded_file = st.file_uploader("上传图片文件 (支持PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

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
                        
                        # 流式输出结果
                        partial_result = ""
                        output_area = st.empty()
                        for piece in stream_inference(img_padded, model):
                            partial_result += piece
                            output_area.markdown(f"```latex\n{partial_result}\n```")
                    
                    st.success("识别完成！")
        except Exception as e:
            st.error(f"处理图片时出错: {e}")
else:
    st.error("模型未能加载，应用无法运行。请检查日志。")