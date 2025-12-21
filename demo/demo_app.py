import gradio as gr
import subprocess
import os
import shutil
import time
import glob

# ==========================================
# ### 🛠️ 配置区域 (Configuration Area)
# ==========================================

# 1. 基础路径设置
# 获取当前 app.py 所在的绝对路径，作为基准
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp_workspace")
os.makedirs(TEMP_DIR, exist_ok=True)

# 2. FFmpeg
FFMPEG_BIN = "ffmpeg"

# 3. YingMusic-SVC (人声分离) 配置
SEPARATION_WORK_DIR = os.path.join(BASE_DIR, "YingMusic-SVC", "accom_separation")
SEPARATION_SCRIPT_NAME = "demo_infer.sh" # 脚本名

# 4. Seed-VC (歌声转换) 配置
SEED_PROJECT_ROOT = os.path.join(BASE_DIR, "seed-vc") # 假设也在同级，如不同请修改
SEED_INFERENCE_SCRIPT = "inference.py"
SEED_PYTHON_EXE = r"/root/anaconda3/envs/seedvc/bin/python" # 请核对你的 python 路径
SEED_CKPT_PATH = os.path.join(SEED_PROJECT_ROOT, "runs", "training-hajimi", "DiT_epoch_00013_step_00500.pth")
SEED_CONFIG_PATH = os.path.join(SEED_PROJECT_ROOT, "runs", "training-hajimi", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml")
USE_F0_CONDITION = "True"


DEFAULT_SONG_PATH = os.path.join(BASE_DIR, "demo", "demo_song.wav") 
DEFAULT_REF_PATH = os.path.join(BASE_DIR, "demo", "demo_ref.wav")
# ==========================================
# ### 🚀 核心功能函数
# ==========================================

def convert_to_mp3(input_path):
    """
    将音频强制转为轻量级 MP3，解决网速慢和格式不兼容问题。
    """
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        return None

    # 生成同名的 mp3 路径
    dir_name = os.path.dirname(input_path)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(dir_name, f"{base_name}_web.mp3")

    print(f">>> 正在压缩音频以加速传输: {output_path}")
    
    try:
        # -b:a 128k 表示比特率 128k（足够听个响，体积非常小）
        subprocess.run([
            FFMPEG_BIN, "-y", 
            "-i", input_path,
            "-ar", "44100", 
            "-ac", "2", 
            "-b:a", "128k", 
            output_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return output_path
    except subprocess.CalledProcessError:
        return input_path # 如果转码失败，勉强返回原文件

def step1_separate(source_audio):
    if not source_audio:
        raise gr.Error("请先在左侧上传原歌曲！")
    
    timestamp = int(time.time())
    # 创建临时文件夹 (注意：这些路径是绝对路径，脚本能识别)
    sep_input_dir = os.path.join(TEMP_DIR, f"sep_input_{timestamp}")
    sep_output_dir = os.path.join(TEMP_DIR, f"sep_output_{timestamp}")
    
    os.makedirs(sep_input_dir, exist_ok=True)
    os.makedirs(sep_output_dir, exist_ok=True)

    # 复制音频
    filename = "input_audio" + os.path.splitext(source_audio)[1]
    shutil.copy(source_audio, os.path.join(sep_input_dir, filename))

    print(f">>> [Step 1] 进入目录: {SEPARATION_WORK_DIR}")
    print(f"    执行脚本: {SEPARATION_SCRIPT_NAME}")

    try:
        # 命令非常简单，直接运行当前目录下的 sh
        cmd = [
            "bash",
            SEPARATION_SCRIPT_NAME, 
            sep_input_dir,  # $1
            sep_output_dir  # $2
        ]
        
        subprocess.run(
            cmd,
            check=True,
            cwd=SEPARATION_WORK_DIR, # 【重点】在这里切换工作目录
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 4. 查找并移动结果
        # BS-Roformer 输出通常带有 _vocals 和 _other 后缀
        print(f">>> 脚本运行完毕，正在查找结果...")
        
        # 使用 glob 搜索任何 wav 文件
        all_wavs = glob.glob(os.path.join(sep_output_dir, "**", "*.wav"), recursive=True)
        
        print(f"DEBUG: 找到的所有音频文件: {all_wavs}")

        if not all_wavs:
             # 如果列表还是空的，说明真的没生成，抛出错误
             raise gr.Error(f"未找到分离文件。目录结构: {os.walk(sep_output_dir)}")

        found_vocal = None
        found_inst = None

        # 遍历找到的文件进行匹配
        for f in all_wavs:
            filename = os.path.basename(f) # 只看文件名
            if "vocals.wav" == filename:
                found_vocal = f
            elif "instrumental.wav" == filename:
                found_inst = f
        
        # 如果还没找到，尝试一种保底逻辑（有的模型输出可能是 input_audio.wav 和 input_audio_music.wav）
        if not found_vocal or not found_inst:
             print("DEBUG: 未能通过文件名关键词匹配，尝试按文件大小排序或直接取前两个...")
             # 这里可以根据你的实际生成结果再调整
             # 假设至少有两个文件，我们尝试强制分配
             if len(all_wavs) >= 2:
                 found_vocal = all_wavs[0]
                 found_inst = all_wavs[1]
             else:
                 raise gr.Error(f"分离结果数量不足。找到的文件: {all_wavs}")

        # 重命名并移动到主 temp 目录
        final_vocal = os.path.join(TEMP_DIR, f"step1_vocal_{timestamp}.wav")
        final_inst = os.path.join(TEMP_DIR, f"step1_inst_{timestamp}.wav")
        
        # 使用 copy 而不是 move，防止跨文件系统错误，且保留原始记录方便调试
        shutil.copy(found_vocal, final_vocal)
        shutil.copy(found_inst, final_inst)
        
        # mp3_vocal = convert_to_mp3(final_vocal)
        # mp3_inst = convert_to_mp3(final_inst)
        
        return final_vocal, final_inst
        # return gr.Audio(value=final_vocal), gr.Audio(value=final_inst)
        

    

    except subprocess.CalledProcessError as e:
        print(f"❌ 错误输出:\n{e.stderr}")
        raise gr.Error(f"分离脚本执行失败 (Exit Code {e.returncode})。请检查控制台日志。")

# ... (Step 2, Step 3 和界面代码保持不变，或者沿用之前的) ...

def step2_convert(vocal_audio, ref_audio):
    """第二步：人声转换 (Vocal -> Converted Vocal) [修正版]"""
    if not vocal_audio:
        raise gr.Error("缺少人声输入！请先完成第一步分离。")
    if not ref_audio:
        raise gr.Error("请在左侧上传目标音色参考音频！")

    timestamp = int(time.time())
    
    # 1. 创建专门的输出目录，而不是传文件路径
    # 脚本会把结果存到这个文件夹里
    vc_output_dir = os.path.join(TEMP_DIR, f"vc_out_{timestamp}")
    os.makedirs(vc_output_dir, exist_ok=True)
    
    print(f">>> [Step 2] 开始转换，输出目录: {vc_output_dir}")

    try:
        subprocess.run(
            [
                SEED_PYTHON_EXE, SEED_INFERENCE_SCRIPT,
                "--source", vocal_audio,
                "--target", ref_audio,
                "--output", vc_output_dir,  # <--- 修改点：这里传入目录
                
                # 【新增】传入自定义模型路径
                "--checkpoint", SEED_CKPT_PATH,
                "--config", SEED_CONFIG_PATH,
                
                # 【新增】传入 F0 参数 (歌声转换建议开启)
                "--f0-condition", USE_F0_CONDITION,
                
                # # 其他可选参数 (你可以根据需要调整)
                # "--diffusion-steps", "30",    # 步数越多质量越好但越慢 (默认30)
                # "--length-adjust", "1.0",     # 语速调整
                # "--inference-cfg-rate", "0.7" # 生成自由度
            ],
            check=True, cwd=SEED_PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # 2. 查找脚本生成的音频文件
        # 脚本生成的文件名通常很长：vc_source_target_....wav
        generated_files = glob.glob(os.path.join(vc_output_dir, "*.wav"))
        
        if not generated_files:
            raise gr.Error(f"转换脚本运行成功，但在目录 {vc_output_dir} 下未找到wav文件。")
            
        # 假设只有一个输出文件，取第一个
        generated_wav = generated_files[0]
        
        # 3. 重命名为我们预期的最终路径
        final_path = os.path.join(TEMP_DIR, f"vc_result_{timestamp}.wav")
        shutil.move(generated_wav, final_path)
        
        print(f">>> 转换成功: {final_path}")
        return final_path
        # return gr.Audio(value=final_path)
        
    except subprocess.CalledProcessError as e:
        print(f"转换错误详情: {e.stderr.decode('utf-8') if e.stderr else '无'}")
        raise gr.Error("转换失败，请查看控制台日志。")

def step3_mix(converted_vocal, original_inst):
    """第三步：混音 (Converted Vocal + Instrument -> Final)"""
    if not converted_vocal:
        raise gr.Error("缺少转换后的人声！请先完成第二步。")
    if not original_inst:
        raise gr.Error("缺少伴奏！请先完成第一步。")
        
    base_name = f"mix_{int(time.time())}"
    final_path = os.path.join(TEMP_DIR, f"{base_name}_final.mp3")
    
    print(f">>> [Step 3] 开始混音...")
    
    try:
        subprocess.run([
            FFMPEG_BIN, "-y",
            "-i", converted_vocal,
            "-i", original_inst,
            "-filter_complex", "amix=inputs=2:duration=longest",
            "-b:a", "1440k",
            final_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return final_path
        # return gr.Audio(value=final_path)
        
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"混音失败: {e}")

# ==========================================
# ### 🖥️ 界面构建 (分步式布局)
# ==========================================
# ==========================================
# ### 🎨 界面美化配置 (UI & CSS)
# ==========================================

# # 1. 自定义 CSS 样式
# custom_css = """
# /* 渐变标题 */
# .gradio-container h1 {
#     background: -webkit-linear-gradient(45deg, #6b21a8, #3b82f6);
#     -webkit-background-clip: text;
#     -webkit-text-fill-color: transparent;
#     font-weight: 800;
#     font-size: 2.5rem !important;
#     text-align: center;
#     margin-bottom: 1rem;
# }

# /* 每一步的卡片样式 */
# .step-card {
#     border: 1px solid #e5e7eb;
#     border-radius: 12px;
#     padding: 15px;
#     background: #ffffff;
#     box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
#     transition: transform 0.2s;
#     margin-bottom: 20px;
# }
# .step-card:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
# }

# /* 步骤编号标题 */
# .step-title {
#     font-size: 1.2rem;
#     font-weight: bold;
#     color: #374151;
#     margin-bottom: 10px;
#     display: flex;
#     align-items: center;
# }
# .step-number {
#     background: #3b82f6;
#     color: white;
#     width: 28px;
#     height: 28px;
#     border-radius: 50%;
#     display: flex;
#     align-items: center;
#     justify-content: center;
#     margin-right: 10px;
#     font-size: 0.9rem;
# }

# /* 按钮美化 */
# button.primary-btn {
#     background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%) !important;
#     border: none !important;
#     color: white !important;
#     font-weight: bold;
# }
# """

# # 2. 创建自定义主题 (基于 Soft 主题微调)
# theme = gr.themes.Soft(
#     primary_hue="indigo",
#     secondary_hue="blue",
#     radius_size="lg",
#     font=[gr.themes.GoogleFont("Noto Sans SC"), "ui-sans-serif", "system-ui", "sans-serif"],
# ).set(
#     button_primary_background_fill="*primary_500",
#     button_primary_background_fill_hover="*primary_600",
# )
simple_css = """
button {
    border-radius: 8px !important; 
}
"""

with gr.Blocks(title="SVC Project Demo", theme=gr.themes.Soft(), css=simple_css) as app:
    gr.Markdown("# 🎵 Hajimi 音乐转换器 (Hajimi Voice Conversion)")
    
    with gr.Row():
        # --- 左侧：原材料区 ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("## 📂 1. 原材料 (Inputs)")
            gr.Markdown("请先在此处上传所需音频。")
            
            src_input = gr.Audio(
                label="原歌曲 (Source Song)", 
                type="filepath",
                value=DEFAULT_SONG_PATH  # <--- 绑定默认歌曲
            )
            
            # 【修改点 2】添加 value=DEFAULT_REF_PATH
            ref_input = gr.Audio(
                label="目标音色 (Target Voice)", 
                type="filepath",
                value=DEFAULT_REF_PATH   # <--- 绑定默认参考音色
            )
            
            gr.Markdown("---")
            gr.Markdown("**说明：**\n左侧准备好后，请按顺序点击右侧的按钮。")

        # --- 右侧：加工流水线 ---
        with gr.Column(scale=2):
            gr.Markdown("## ⚙️ 2. 工作流程 (Processing Pipeline)")
            
            # === 第一步 ===
            with gr.Group():
                gr.Markdown("### Step 1: 人声分离 (Separation)")
                btn_step1 = gr.Button("👇 点击执行分离 Click to separate", variant="primary")
                with gr.Row():
                    # 这两个组件既是第一步的输出，也是后续步骤的输入来源
                    out_vocal = gr.Audio(label="分离结果：纯人声 (Vocal)", type="filepath", interactive=False)
                    out_inst = gr.Audio(label="分离结果：纯伴奏 (Instrumental)", type="filepath", interactive=False)

            # === 第二步 ===
            with gr.Group():
                gr.Markdown("### Step 2: 歌声转换 (Voice Conversion via seed-vc)  ")
                btn_step2 = gr.Button("👇 点击执行转换 Click to convert", variant="primary")
                # 这是第二步的输出
                out_converted = gr.Audio(label="转换结果：新的人声 (Converted)", type="filepath", interactive=False)

            # === 第三步 ===
            with gr.Group():
                gr.Markdown("### Step 3: 最终合成 (Mixing)")
                btn_step3 = gr.Button("👇 点击执行合成 Click to mix", variant="primary")
                # 这是最终输出
                out_final = gr.Audio(label="🎉 最终成品 (Final Song)", type="filepath")

    # ==========================================
    # ### 🔗 逻辑绑定 (Data Flow)
    # ==========================================
    
    # 点击 Step 1 按钮 -> 读取左侧原曲 -> 输出 Vocal 和 Inst
    btn_step1.click(
        fn=step1_separate,
        inputs=[src_input],
        outputs=[out_vocal, out_inst]
    )
    
    # 点击 Step 2 按钮 -> 读取 Step 1 的 Vocal 和 左侧的目标音色 -> 输出 Converted Vocal
    btn_step2.click(
        fn=step2_convert,
        inputs=[out_vocal, ref_input],
        outputs=[out_converted]
    )
    
    # 点击 Step 3 按钮 -> 读取 Step 2 的 Converted Vocal 和 Step 1 的 Inst -> 输出 Final
    btn_step3.click(
        fn=step3_mix,
        inputs=[out_converted, out_inst],
        outputs=[out_final]
    )

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0")

# ==========================================
# ### 🖥️ 界面构建 (美化版)
# ==========================================

# with gr.Blocks(title="AI 歌声转换工作台", theme=theme, css=custom_css) as app:
    
#     # --- 顶部标题区 ---
#     gr.Markdown("# 🎵 AI 歌声转换工作台 (SVC Studio)")
#     gr.Markdown("#### 🚀 基于 YingMusic-SVC & Seed-VC 的全流程处理流水线")

#     # --- 状态指示器 ---
#     with gr.Row():
#         gr.Markdown(
#             """
#             <div style="text-align: center; font-size: 1.1em; color: #666;">
#             1️⃣ <b>人声分离</b> <span style="color:#ccc">──────▶</span> 
#             2️⃣ <b>歌声转换</b> <span style="color:#ccc">──────▶</span> 
#             3️⃣ <b>最终混音</b>
#             </div>
#             """
#         )

#     with gr.Row():
        
#         # ==================================
#         # ⬅️ 左侧：控制面板 (输入 + 设置)
#         # ==================================
#         with gr.Column(scale=1, variant="panel"):
#             gr.Markdown("### 🎛️ 控制面板")
            
#             with gr.Group():
#                 gr.Markdown("**1. 上传素材**")
#                 # 使用 value 绑定我们在配置区定义的默认路径 
#                 src_input = gr.Audio(label="🎵 原歌曲 (含伴奏)", type="filepath", value=DEFAULT_SONG_PATH)
#                 ref_input = gr.Audio(label="🗣️ 目标音色 (说话人)", type="filepath", value=DEFAULT_REF_PATH)
            
#             gr.Markdown("---")
            
#             # --- 高级设置 (折叠起来不占地方) ---
#             with gr.Accordion("🛠️ 模型与高级设置", open=False):
#                 gr.Markdown(f"**Seed-VC 模型:**\n`{os.path.basename(SEED_CKPT_PATH)}`")
#                 gr.Markdown(f"**F0 模式:** `{USE_F0_CONDITION}`")
#                 gr.Markdown(f"**分离脚本:** `{SEPARATION_SCRIPT_NAME}`")
#                 gr.Markdown("**注意：** 修改模型请直接编辑 `demo_app.py` 顶部的配置区域。")

#         # ==================================
#         # ➡️ 右侧：执行流水线
#         # ==================================
#         with gr.Column(scale=2):
            
#             # --- Step 1: 分离 ---
#             with gr.Group(elem_classes="step-card"):
#                 gr.HTML('<div class="step-title"><div class="step-number">1</div> 人声分离 (Separation)</div>')
#                 gr.Markdown("将原歌曲拆分为 **纯人声** 和 **纯伴奏**。")
                
#                 btn_step1 = gr.Button("开始分离 (Start Separation)", variant="primary", elem_classes="primary-btn")
                
#                 with gr.Row():
#                     out_vocal = gr.Audio(label="分离结果：人声 (Vocal)", type="filepath", interactive=False, show_share_button=False)
#                     out_inst = gr.Audio(label="分离结果：伴奏 (Inst)", type="filepath", interactive=False, show_share_button=False)

#             # --- Step 2: 转换 ---
#             with gr.Group(elem_classes="step-card"):
#                 gr.HTML('<div class="step-title"><div class="step-number">2</div> 歌声转换 (Conversion)</div>')
#                 gr.Markdown("使用 **Seed-VC** 将分离出的人声转换为目标音色。")
                
#                 btn_step2 = gr.Button("开始转换 (Start Conversion)", variant="primary", elem_classes="primary-btn")
                
#                 out_converted = gr.Audio(label="转换结果：新的人声 (Converted)", type="filepath", interactive=False, show_share_button=False)

#             # --- Step 3: 混音 ---
#             with gr.Group(elem_classes="step-card"):
#                 gr.HTML('<div class="step-title"><div class="step-number">3</div> 最终合成 (Mixing)</div>')
#                 gr.Markdown("将 **新的人声** 与 **第一步的伴奏** 重新混合。")
                
#                 btn_step3 = gr.Button("开始混音 (Final Mix)", variant="primary", elem_classes="primary-btn")
                
#                 out_final = gr.Audio(label="🎉 最终成品 (Final Song)", type="filepath", show_download_button=True)

#     # ==========================================
#     # ### 🔗 逻辑绑定 (保持不变)
#     # ==========================================
    
#     btn_step1.click(fn=step1_separate, inputs=[src_input], outputs=[out_vocal, out_inst])
#     btn_step2.click(fn=step2_convert, inputs=[out_vocal, ref_input], outputs=[out_converted])
#     btn_step3.click(fn=step3_mix, inputs=[out_converted, out_inst], outputs=[out_final])

# # 启动 (自动寻找空闲端口)
# if __name__ == "__main__":
#     app.queue().launch(server_name="0.0.0.0", show_error=True)