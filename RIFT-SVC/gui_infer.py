import numpy as np
import torch
import torchaudio
import click
import gradio as gr
import tempfile
import gc
import traceback
import os
from slicer import Slicer

from infer import (
    load_models, 
    load_audio, 
    apply_fade, 
    batch_process_segments
)

global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, device
svc_model = vocoder = rmvpe = hubert = rms_extractor = spk2idx = dataset_cfg = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LANGUAGES = {
    "En": {
        "app_title": "RIFT-SVC Voice Conversion",
        "main_title": "🎤 RIFT-SVC Voice Conversion",
        "subtitle": "Convert singing or speech to target voice using RIFT-SVC model",
        "github_info": "🔗 <strong>Want to fine-tune your own speakers?</strong> Visit the <a href=\"https://github.com/Pur1zumu/RIFT-SVC\" target=\"_blank\">RIFT-SVC GitHub repository</a> for complete training and fine-tuning guides.",
        "audio_note": "📝 <strong>Note:</strong> For best results, use clean audio with minimal background noise.",
        
        "input_section": "### 📥 Input",
        "model_path_label": "Model Path",
        "model_path_placeholder": "Enter your model file path",
        "load_model_btn": "🔄 Load Model",
        "input_audio_label": "Input Audio File",
        
        "basic_params": "⚙️ Basic Parameters",
        "target_speaker": "Target Speaker",
        "key_shift": "Key Shift (semitones)",
        "infer_steps": "Inference Steps",
        "infer_steps_info": "Lower = faster but lower quality, Higher = slower but better quality",
        "use_fp16": "Use FP16 Precision",
        "use_fp16_info": "Enable for better performance, may reduce precision on some GPUs",
        "pitch_filter": "Pitch Filter",
        "pitch_filter_info": "0=None, 1=Light filtering, 2=Strong filtering (helps with broken/choppy sounds)",
        "batch_size": "Batch Size",
        "batch_size_info": "Number of segments to process in parallel. Higher values can be faster on GPUs with sufficient memory.",
        
        "adv_cfg_params": "🔬 Advanced CFG Parameters",
        "ds_cfg_strength": "Content Vector Guidance Strength",
        "ds_cfg_strength_info": "Higher values can improve content preservation and pronunciation clarity. Too high will be overkill.",
        "spk_cfg_strength": "Speaker Guidance Strength",
        "spk_cfg_strength_info": "Higher values can enhance speaker similarity. Too high may cause distortion.",
        "skip_cfg_strength": "Layer Guidance Strength (Experimental)",
        "skip_cfg_strength_info": "Enhances feature rendering at specified layers. Effect depends on target layer functionality.",
        "cfg_skip_layers": "CFG Skip Layers (Experimental)",
        "cfg_skip_layers_info": "Target enhancement layer index, -1 to disable this feature",
        "cfg_rescale": "CFG Rescale Factor",
        "cfg_rescale_info": "Constrains overall guidance strength. Increase when guidance effects are too strong.",
        "cvec_downsample": "Content Vector Downsample Rate for Reverse Guidance",
        "cvec_downsample_info": "Higher values (may) improve content clarity.",
        
        "slicer_params": "✂️ Slicer Parameters",
        "slicer_threshold": "Threshold (dB)",
        "slicer_threshold_info": "Silence detection threshold",
        "slicer_min_length": "Minimum Length (ms)",
        "slicer_min_length_info": "Minimum segment length",
        "slicer_min_interval": "Minimum Silence Interval (ms)",
        "slicer_min_interval_info": "Minimum interval for segment splitting",
        "slicer_hop_size": "Hop Size (ms)",
        "slicer_hop_size_info": "Window size for segment detection",
        "slicer_max_sil": "Maximum Silence Kept (ms)",
        "slicer_max_sil_info": "Maximum silence length kept at each segment edge",
        
        "convert_btn": "🎵 Convert Voice",
        "output_section": "### 📤 Output",
        "output_audio_label": "Converted Audio",
        "init_message": "⏳ Please load a model to begin.",
        
        "quick_tips": "🔍 Quick Tips",
        "tips_content": """
                    <ul>
                        <li><strong>Key Shift:</strong> Adjust pitch up or down in semitones.</li>
                        <li><strong>Inference Steps:</strong> More steps = better quality but slower.</li>
                        <li><strong>Pitch Filter:</strong> Helps with pitch stability in challenging audio.</li>
                        <li><strong>CFG Parameters:</strong> Adjust conversion quality and timbre.</li>
                    </ul>
                """,
        
        "processing": "⏳ Processing... Please wait.",
        "loading_audio": "Processing: Loading audio...",
        "slicing_audio": "Processing: Slicing audio...",
        "start_conversion": "Processing: Starting conversion...",
        "processing_segment": "Processing: Segment {}/{}",
        "finalizing_audio": "Processing: Finalizing audio...",
        "processing_complete": "Processing complete!",
        
        "conversion_complete": "✅ Conversion complete! Converted to **{}** with **{}** semitone shift.",
        "error_no_audio": "❌ Error: No input audio provided.",
        "error_no_model": "❌ Error: Model not loaded. Please load a model first.",
        "error_invalid_speaker": "❌ Error: Invalid speaker selection. Available speakers: {}",
        "error_no_segments": "❌ Error: No valid audio segments found in the input file.",
        "error_out_of_memory": "❌ Error: Out of memory. Try a shorter audio file or reduce inference steps.",
        "error_conversion": "❌ Error during conversion: {}",
        "error_details": "❌ Error during conversion: {}\n\nDetails: {}",
        "error_model_not_found": "❌ Error: Model not found",
        "model_loaded_success": "✅ Model loaded successfully! Available speakers: ",
        "error_loading_model": "❌ Error: Failed to load model",
        "error_details_label": "Error Details"
    },
    "中文": {
        "app_title": "RIFT-SVC 声音转换",
        "main_title": "🎤 RIFT-SVC 歌声音色转换",
        "subtitle": "使用 RIFT-SVC 模型将歌声或语音转换为目标音色",
        "github_info": "🔗 <strong>想要微调自己的说话人？</strong> 请访问 <a href=\"https://github.com/Pur1zumu/RIFT-SVC\" target=\"_blank\">RIFT-SVC GitHub 仓库</a> 获取完整的训练和微调指南。",
        "audio_note": "📝 <strong>注意：</strong> 为获得最佳效果，请使用背景噪音较少的干净音频。",
        
        "input_section": "### 📥 输入",
        "model_path_label": "模型路径",
        "model_path_placeholder": "请输入您的模型文件路径",
        "load_model_btn": "🔄 加载模型",
        "input_audio_label": "输入音频文件",
        
        "basic_params": "⚙️ 基本参数",
        "target_speaker": "目标说话人",
        "key_shift": "音调调整（半音）",
        "infer_steps": "推理步数",
        "infer_steps_info": "更低的值 = 更快但质量较低，更高的值 = 更慢但质量更好",
        "use_fp16": "使用 FP16 精度",
        "use_fp16_info": "启用以提高性能，在某些GPU上可能会降低精度",
        "pitch_filter": "音高滤波",
        "pitch_filter_info": "0=无，1=轻度过滤，2=强力过滤（有助于解决断音/破音问题）",
        "batch_size": "批量大小",
        "batch_size_info": "并行处理段落的数量。更高的值可以在具有足够内存的GPU上更快。",
        
        "adv_cfg_params": "🔬 高级CFG参数",
        "ds_cfg_strength": "内容向量引导强度",
        "ds_cfg_strength_info": "更高的值可以改善内容保留和咬字清晰度。过高会用力过猛。",
        "spk_cfg_strength": "说话人引导强度",
        "spk_cfg_strength_info": "更高的值可以增强说话人相似度。过高可能导致音色失真。",
        "skip_cfg_strength": "层引导强度（实验性功能）",
        "skip_cfg_strength_info": "增强指定层的特征渲染。效果取决于目标层的功能。",
        "cfg_skip_layers": "CFG跳过层（实验性功能）",
        "cfg_skip_layers_info": "目标增强层下标，-1为禁用此功能",
        "cfg_rescale": "CFG重缩放因子",
        "cfg_rescale_info": "约束整体引导强度。当引导效果过于强烈时使用调高该值。",
        "cvec_downsample": "用于反向引导的内容向量下采样率",
        "cvec_downsample_info": "更高的值（可能）可以提高内容清晰度。",
        
        "slicer_params": "✂️ 切片参数",
        "slicer_threshold": "阈值 (dB)",
        "slicer_threshold_info": "静音检测阈值",
        "slicer_min_length": "最小长度 (毫秒)",
        "slicer_min_length_info": "最小片段长度",
        "slicer_min_interval": "最小静音间隔 (毫秒)",
        "slicer_min_interval_info": "分割片段的最小间隔",
        "slicer_hop_size": "跳跃大小 (毫秒)",
        "slicer_hop_size_info": "片段检测窗口大小",
        "slicer_max_sil": "保留的最大静音 (毫秒)",
        "slicer_max_sil_info": "保留在每个片段边缘的最大静音长度",
        
        "convert_btn": "🎵 转换声音",
        "output_section": "### 📤 输出",
        "output_audio_label": "转换后的音频",
        "init_message": "⏳ 请加载模型以开始使用。",
        
        "quick_tips": "🔍 快速提示",
        "tips_content": """
                    <ul>
                        <li><strong>音调调整：</strong> 以半音为单位上调或下调音高。</li>
                        <li><strong>推理步骤：</strong> 步骤越多 = 质量越好但速度越慢。</li>
                        <li><strong>音高滤波：</strong> 有助于提高具有挑战性的音频中的音高稳定性。</li>
                        <li><strong>CFG参数：</strong> 调整转换质量和音色。</li>
                    </ul>
                """,
        
        "processing": "⏳ 处理中... 请稍候。",
        "loading_audio": "处理中: 加载音频...",
        "slicing_audio": "处理中: 切分音频...",
        "start_conversion": "处理中: 开始转换...",
        "processing_segment": "处理中: 片段 {}/{}",
        "finalizing_audio": "处理中: 完成音频...",
        "processing_complete": "处理完成!",
        
        "conversion_complete": "✅ 转换完成! 已转换为 **{}** 并调整 **{}** 个半音。",
        "error_no_audio": "❌ 错误: 未提供输入音频。",
        "error_no_model": "❌ 错误: 模型未加载。请先加载模型。",
        "error_invalid_speaker": "❌ 错误: 无效的说话人选择。可用说话人: {}",
        "error_no_segments": "❌ 错误: 在输入文件中未找到有效的音频片段。",
        "error_out_of_memory": "❌ 错误: 内存不足。请尝试更短的音频文件或减少推理步骤。",
        "error_conversion": "❌ 转换过程中出错: {}",
        "error_details": "❌ 转换过程中出错: {}\n\n详细信息: {}",
        "error_model_not_found": "❌ 错误: 找不到模型文件",
        "model_loaded_success": "✅ 模型加载成功！可用说话人: ",
        "error_loading_model": "❌ 错误: 加载模型出错",
        "error_details_label": "错误详细信息"
    }
}

current_language = "En"

def initialize_models(model_path):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, current_language
    
    lang = LANGUAGES[current_language]
    use_fp16 = True
    
    if svc_model is not None:
        del svc_model
        del vocoder
        del rmvpe
        del hubert
        del rms_extractor
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        if not os.path.exists(model_path):
            return [], f"{lang['error_model_not_found']}: {model_path}"
        
        svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg = load_models(model_path, device, use_fp16)
        available_speakers = list(spk2idx.keys())
        return available_speakers, f"{lang['model_loaded_success']} {', '.join(available_speakers)}"
    except Exception as e:
        error_trace = traceback.format_exc()
        return [], f"{lang['error_loading_model']}: {str(e)}\n\n{lang['error_details_label']}: {error_trace}"

def process_with_progress(
    progress=gr.Progress(),
    input_audio=None,
    speaker=None,
    key_shift=0,
    infer_steps=32,
    robust_f0=1,
    use_fp16=True,
    batch_size=1,
    ds_cfg_strength=0.1,
    spk_cfg_strength=1.0,
    skip_cfg_strength=0.0,
    cfg_skip_layers=6,
    cfg_rescale=0.7,
    cvec_downsample_rate=2,
    slicer_threshold=-30.0,
    slicer_min_length=3000,
    slicer_min_interval=100,
    slicer_hop_size=10,
    slicer_max_sil_kept=200
):
    global svc_model, vocoder, rmvpe, hubert, rms_extractor, spk2idx, dataset_cfg, current_language
    
    lang = LANGUAGES[current_language]
    
    target_loudness = -18.0
    restore_loudness = True
    fade_duration = 20.0
    
    if input_audio is None:
        return None, lang["error_no_audio"]
    
    if svc_model is None:
        return None, lang["error_no_model"]
    
    if speaker is None or speaker not in spk2idx:
        return None, lang["error_invalid_speaker"].format(", ".join(spk2idx.keys()))
    
    try:
        progress(0, desc=lang["loading_audio"])
        
        speaker_id = spk2idx[speaker]
        
        hop_length = 512
        sample_rate = 44100
        
        if cfg_skip_layers < 0:
            cfg_skip_layers_value = None
        else:
            cfg_skip_layers_value = cfg_skip_layers
        
        audio = load_audio(input_audio, sample_rate)
        
        slicer = Slicer(
            sr=sample_rate,
            threshold=slicer_threshold,
            min_length=slicer_min_length,
            min_interval=slicer_min_interval,
            hop_size=slicer_hop_size,
            max_sil_kept=slicer_max_sil_kept
        )
        
        progress(0.1, desc=lang["slicing_audio"])
        segments_with_pos = slicer.slice(audio)
        
        if not segments_with_pos:
            return None, lang["error_no_segments"]
        
        fade_samples = int(fade_duration * sample_rate / 1000)
        
        progress(0.2, desc=lang["start_conversion"])
        
        with torch.no_grad():
            processed_segments = batch_process_segments(
                segments_with_pos, svc_model, vocoder, rmvpe, hubert, rms_extractor,
                speaker_id, sample_rate, hop_length, device,
                key_shift, infer_steps, ds_cfg_strength, spk_cfg_strength,
                skip_cfg_strength, cfg_skip_layers_value, cfg_rescale,
                cvec_downsample_rate, target_loudness, restore_loudness,
                robust_f0, use_fp16, batch_size, progress, lang["processing_segment"]
            )
            
            result_audio = np.zeros(len(audio) + fade_samples)
            
            for idx, (start_sample, audio_out, expected_length) in enumerate(processed_segments):
                segment_progress = 0.8 + (0.1 * (idx / len(processed_segments)))
                progress(segment_progress, desc=lang["finalizing_audio"])
                
                if len(audio_out) > expected_length:
                    audio_out = audio_out[:expected_length]
                elif len(audio_out) < expected_length:
                    audio_out = np.pad(audio_out, (0, expected_length - len(audio_out)), 'constant')
                
                if idx > 0:
                    audio_out = apply_fade(audio_out.copy(), fade_samples, fade_in=True)
                    result_audio[start_sample:start_sample + fade_samples] *= np.linspace(1, 0, fade_samples)
                
                if idx < len(processed_segments) - 1:
                    audio_out[-fade_samples:] *= np.linspace(1, 0, fade_samples)
                
                result_audio[start_sample:start_sample + len(audio_out)] += audio_out
        
        progress(0.9, desc=lang["finalizing_audio"])
        result_audio = result_audio[:len(audio)]
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            output_path = temp_file.name
        
        torchaudio.save(output_path, torch.from_numpy(result_audio).unsqueeze(0).float(), sample_rate)
        
        progress(1.0, desc=lang["processing_complete"])
        return (sample_rate, result_audio), lang["conversion_complete"].format(speaker, key_shift)
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            gc.collect()
            
            return None, lang["error_out_of_memory"]
        else:
            return None, lang["error_conversion"].format(str(e))
    except Exception as e:
        error_trace = traceback.format_exc()
        return None, lang["error_details"].format(str(e), error_trace)
    finally:
        torch.cuda.empty_cache()
        gc.collect()

def create_ui():
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        position: relative;
    }
    .container {
        max-width: 1200px;
        margin: auto;
    }
    .footer {
        margin-top: 20px;
        text-align: center;
        font-size: 0.9em;
        color: #666;
    }
    .title {
        text-align: center;
        margin-bottom: 10px;
        padding-top: 10px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 20px;
        color: #666;
    }
    .button-primary {
        background-color: #5460DE !important;
    }
    .output-message {
        margin-top: 10px;
        padding: 10px;
        border-radius: 4px;
        background-color: #f8f9fa;
        border-left: 4px solid #5460DE;
    }
    .error-message {
        color: #d62828;
        font-weight: bold;
    }
    .success-message {
        color: #588157;
        font-weight: bold;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #5460DE;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    .lang-container {
        position: absolute;
        top: 10px;
        right: 20px;
        z-index: 1000;
        width: auto;
        max-width: 180px;
    }
    .compact-dropdown .wrap .label-wrap {
        font-size: 0.9em !important;
    }
    .compact-dropdown .wrap {
        max-width: 150px !important;
    }
    """
    
    available_speakers = []
    global current_language
    lang = LANGUAGES[current_language]
    init_message = lang["init_message"]
    
    with gr.Blocks(css=css, theme=gr.themes.Soft(), title=lang["app_title"]) as app:
        with gr.Row(elem_classes="lang-container"):
            language_selector = gr.Dropdown(
                choices=["En", "中文"], 
                value=current_language, 
                label="Language / 语言",
                elem_classes="compact-dropdown"
            )
        
        html_header = gr.HTML(f"""
        <div class="title">
            <h1>{lang["main_title"]}</h1>
        </div>
        <div class="info-box">
            <p>{lang["audio_note"]}</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    input_markdown = gr.Markdown(lang["input_section"])
                    model_path = gr.Textbox(label=lang["model_path_label"], value="", placeholder=lang["model_path_placeholder"], interactive=True)
                    reload_btn = gr.Button(lang["load_model_btn"], elem_id="reload_btn")
                    input_audio = gr.Audio(label=lang["input_audio_label"], type="filepath", elem_id="input_audio")
                
                with gr.Accordion(lang["basic_params"], open=True) as basic_params_accordion:
                    speaker = gr.Dropdown(label=lang["target_speaker"], interactive=True, elem_id="speaker")
                    key_shift = gr.Slider(minimum=-12, maximum=12, step=1, value=0, label=lang["key_shift"], elem_id="key_shift")
                    infer_steps = gr.Slider(minimum=8, maximum=64, step=1, value=32, label=lang["infer_steps"], elem_id="infer_steps", 
                                           info=lang["infer_steps_info"])
                    use_fp16 = gr.Checkbox(label=lang["use_fp16"], value=True, info=lang["use_fp16_info"], elem_id="use_fp16")
                    robust_f0 = gr.Radio(choices=[0, 1, 2], value=1, label=lang["pitch_filter"], 
                                        info=lang["pitch_filter_info"], 
                                        elem_id="robust_f0")
                    batch_size = gr.Slider(minimum=1, maximum=64, step=1, value=1, label=lang["batch_size"], info=lang["batch_size_info"], elem_id="batch_size")
                
                with gr.Accordion(lang["adv_cfg_params"], open=True) as adv_cfg_accordion:
                    ds_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.1, 
                                               label=lang["ds_cfg_strength"], 
                                               info=lang["ds_cfg_strength_info"], 
                                               elem_id="ds_cfg_strength")
                    spk_cfg_strength = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=1.0, 
                                                label=lang["spk_cfg_strength"], 
                                                info=lang["spk_cfg_strength_info"], 
                                                elem_id="spk_cfg_strength")
                    skip_cfg_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, 
                                                 label=lang["skip_cfg_strength"], 
                                                 info=lang["skip_cfg_strength_info"], 
                                                 elem_id="skip_cfg_strength")
                    cfg_skip_layers = gr.Number(value=-1, label=lang["cfg_skip_layers"], precision=0, 
                                               info=lang["cfg_skip_layers_info"], 
                                               elem_id="cfg_skip_layers")
                    cfg_rescale = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.7, 
                                           label=lang["cfg_rescale"], 
                                           info=lang["cfg_rescale_info"], 
                                           elem_id="cfg_rescale")
                    cvec_downsample_rate = gr.Radio(choices=[1, 2, 4, 8], value=2, 
                                                  label=lang["cvec_downsample"], 
                                                  info=lang["cvec_downsample_info"], 
                                                  elem_id="cvec_downsample_rate")
                
                with gr.Accordion(lang["slicer_params"], open=False) as slicer_accordion:
                    slicer_threshold = gr.Slider(minimum=-60.0, maximum=-20.0, step=0.1, value=-30.0, 
                                                label=lang["slicer_threshold"], 
                                                info=lang["slicer_threshold_info"], 
                                                elem_id="slicer_threshold")
                    slicer_min_length = gr.Slider(minimum=1000, maximum=10000, step=100, value=3000, 
                                                 label=lang["slicer_min_length"], 
                                                 info=lang["slicer_min_length_info"], 
                                                 elem_id="slicer_min_length")
                    slicer_min_interval = gr.Slider(minimum=10, maximum=500, step=10, value=100, 
                                                   label=lang["slicer_min_interval"], 
                                                   info=lang["slicer_min_interval_info"], 
                                                   elem_id="slicer_min_interval")
                    slicer_hop_size = gr.Slider(minimum=1, maximum=20, step=1, value=10, 
                                              label=lang["slicer_hop_size"], 
                                              info=lang["slicer_hop_size_info"], 
                                              elem_id="slicer_hop_size")
                    slicer_max_sil_kept = gr.Slider(minimum=10, maximum=1000, step=10, value=200, 
                                                  label=lang["slicer_max_sil"], 
                                                  info=lang["slicer_max_sil_info"], 
                                                  elem_id="slicer_max_sil_kept")
            
            with gr.Column(scale=1):
                convert_btn = gr.Button(lang["convert_btn"], variant="primary", elem_id="convert_btn")
                output_markdown = gr.Markdown(lang["output_section"])
                output_audio = gr.Audio(label=lang["output_audio_label"], elem_id="output_audio", autoplay=False, show_share_button=False)
                output_message = gr.Markdown(init_message, elem_id="output_message", elem_classes="output-message")
                
                tips_html = gr.HTML(f"""
                <div class="info-box">
                    <h4>{lang["quick_tips"]}</h4>
                    {lang["tips_content"]}
                </div>
                """)
        
        def update_language(selected_language):
            global current_language
            current_language = selected_language
            lang = LANGUAGES[current_language]
            
            return [
                gr.update(label=lang["model_path_label"], placeholder=lang["model_path_placeholder"]),
                gr.update(value=lang["load_model_btn"]),
                gr.update(label=lang["input_audio_label"]),
                
                gr.update(label=lang["target_speaker"]),
                gr.update(label=lang["key_shift"]),
                gr.update(label=lang["infer_steps"], info=lang["infer_steps_info"]),
                gr.update(label=lang["use_fp16"], info=lang["use_fp16_info"]),
                gr.update(label=lang["pitch_filter"], info=lang["pitch_filter_info"]),
                gr.update(label=lang["batch_size"], info=lang["batch_size_info"]),
                
                gr.update(label=lang["basic_params"]),
                gr.update(label=lang["adv_cfg_params"]),
                gr.update(label=lang["slicer_params"]),
                
                gr.update(label=lang["ds_cfg_strength"], info=lang["ds_cfg_strength_info"]),
                gr.update(label=lang["spk_cfg_strength"], info=lang["spk_cfg_strength_info"]),
                gr.update(label=lang["skip_cfg_strength"], info=lang["skip_cfg_strength_info"]),
                gr.update(label=lang["cfg_skip_layers"], info=lang["cfg_skip_layers_info"]),
                gr.update(label=lang["cfg_rescale"], info=lang["cfg_rescale_info"]),
                gr.update(label=lang["cvec_downsample"], info=lang["cvec_downsample_info"]),
                
                gr.update(label=lang["slicer_threshold"], info=lang["slicer_threshold_info"]),
                gr.update(label=lang["slicer_min_length"], info=lang["slicer_min_length_info"]),
                gr.update(label=lang["slicer_min_interval"], info=lang["slicer_min_interval_info"]),
                gr.update(label=lang["slicer_hop_size"], info=lang["slicer_hop_size_info"]),
                gr.update(label=lang["slicer_max_sil"], info=lang["slicer_max_sil_info"]),
                
                gr.update(value=lang["convert_btn"]),
                gr.update(value=lang["output_section"]),
                gr.update(label=lang["output_audio_label"]),
                gr.update(value=lang["init_message"]),
                
                gr.update(value=f"""
                <div class="title">
                    <h1>{lang["main_title"]}</h1>
                </div>
                <div class="subtitle">
                    <h3>{lang["subtitle"]}</h3>
                </div>
                <div class="info-box">
                    <p>{lang["github_info"]}</p>
                </div>
                <div class="info-box">
                    <p>{lang["audio_note"]}</p>
                </div>
                """),
                
                gr.update(value=f"""
                <div class="info-box">
                    <h4>{lang["quick_tips"]}</h4>
                    {lang["tips_content"]}
                </div>
                """)
            ]
        
        def load_model_and_update_speakers(model_path):
            available_speakers, message = initialize_models(model_path)
            
            if available_speakers and len(available_speakers) > 0:
                return gr.update(choices=available_speakers, value=available_speakers[0]), message
            else:
                return gr.update(choices=[], value=None), message
        
        language_selector.change(
            fn=update_language,
            inputs=[language_selector],
            outputs=[
                model_path, reload_btn, input_audio,  
                speaker, key_shift, infer_steps, use_fp16, robust_f0, batch_size,  
                basic_params_accordion, adv_cfg_accordion, slicer_accordion,  
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength, 
                cfg_skip_layers, cfg_rescale, cvec_downsample_rate,
                slicer_threshold, slicer_min_length, slicer_min_interval,
                slicer_hop_size, slicer_max_sil_kept,
                convert_btn, output_markdown, output_audio, output_message,  
                html_header, tips_html
            ]
        )
        
        reload_btn.click(
            fn=load_model_and_update_speakers,
            inputs=[model_path],
            outputs=[speaker, output_message]
        )
        
        convert_btn.click(
            fn=lambda: LANGUAGES[current_language]["processing"],
            inputs=None,
            outputs=output_message,
            queue=False
        ).then(
            fn=process_with_progress,
            inputs=[
                input_audio, speaker, key_shift, infer_steps, robust_f0, use_fp16,
                batch_size,
                ds_cfg_strength, spk_cfg_strength, skip_cfg_strength, cfg_skip_layers, cfg_rescale, cvec_downsample_rate,
                slicer_threshold, slicer_min_length, slicer_min_interval, slicer_hop_size, slicer_max_sil_kept
            ],
            outputs=[output_audio, output_message],
            show_progress_on=output_audio
        )
    
    return app

@click.command()
@click.option('--share', is_flag=True, help='Share the app')
@click.option('--language', default='En', help='Default language (en or 中文)')
def main(share=False, language='En'):
    global current_language
    if language in LANGUAGES:
        current_language = language
    else:
        current_language = 'En'
    
    app = create_ui()
    app.launch(share=share)

if __name__ == "__main__":
    main()