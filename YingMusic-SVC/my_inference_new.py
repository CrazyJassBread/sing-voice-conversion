import os
import numpy as np
import shutil
import warnings
import argparse
import torch
import yaml
import time
import torchaudio
import librosa

from modules.commons import *
from Remix.auger import echo_then_reverb_save
from mm4 import preprocess_voice_conversion
from hf_utils import load_custom_model_from_hf
from modules.adapters import AdaptedCFM

warnings.simplefilter('ignore')
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'

########## tools ##########
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2

def load_models_api(args, device=torch.device("cuda")):
    dit_checkpoint_path = args.checkpoint
    print(f'load model from {dit_checkpoint_path}')
    dit_config_path = args.config
    print(f'load config from {dit_config_path}')
    
    # f0 extractor
    from modules.rmvpe import RMVPE
    model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
    f0_extractor = RMVPE(model_path, is_half=False, device=device)
    f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    
    # Build base model
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Wrap with AdaptedCFM if requested
    if args.use_lora or args.use_rva:
        print("Wrapping model with AdaptedCFM...")
        lora_config = {}
        if args.use_lora:
            # Read from config with fallback defaults
            sfa_cfg = config.get("model_params", {}).get("sfa_lora", {})
            lora_config = {
                'rank': sfa_cfg.get('rank', 8),
                'alpha': sfa_cfg.get('alpha', 16.0),
                'num_freq_bins': config["preprocess_params"]["spect_params"]["n_mels"],
                'dropout': 0.0,  # No dropout during inference
                'use_freq_gate': True,
                'target_modules': tuple(sfa_cfg.get('target_modules', ['wqkv', 'wo'])),
                'sr': sr,
                'gate_mode': sfa_cfg.get('gate_mode', 'feature_wise'),
            }
        
        rva_config = {}
        if args.use_rva:
            # Read from config with fallback defaults
            rva_cfg = config.get("model_params", {}).get("rva", {})
            rva_config = {
                'in_channels': model_params.DiT.in_channels,
                'hidden_channels': rva_cfg.get('hidden_channels', 256),
                'style_dim': model_params.style_encoder.dim,
                'cond_dim': model_params.length_regulator.channels,
                'use_unet': rva_cfg.get('use_unet', True),
                'guidance_schedule': rva_cfg.get('guidance_schedule', 'cosine'),
                'alpha_min': rva_cfg.get('alpha_min', 0.0),
                'alpha_max': rva_cfg.get('alpha_max', 0.5),
            }
        
        # We use 'full' mode here just to ensure all params are registered, 
        # though we only need eval
        model.cfm = AdaptedCFM(
            model.cfm,
            lora_config=lora_config,
            rva_config=rva_config,
            training_mode='full', 
            freeze_base=False
        )

    # Load checkpoints with smart key mapping
    print(f"Loading checkpoint from {dit_checkpoint_path}")
    state_dict = torch.load(dit_checkpoint_path, map_location="cpu")
    
    # Handle 'net' key if present (common in training checkpoints)
    if "net" in state_dict:
        state_dict = state_dict["net"]
    
    # Prepare model state dict
    model_state = model.cfm.state_dict() if hasattr(model, 'cfm') else {}
    
    # Check if we need to remap keys for AdaptedCFM
    is_adapted = isinstance(model.cfm, AdaptedCFM) if hasattr(model, 'cfm') else False
    
    if is_adapted and 'cfm' in state_dict:
        cfm_state = state_dict['cfm']
        new_cfm_state = {}
        
        # Check if checkpoint is from a standard model (keys start with 'estimator.')
        # but our model is adapted (expects 'base_cfm.estimator.')
        has_base_prefix = any(k.startswith('base_cfm.') for k in cfm_state.keys())
        
        if not has_base_prefix:
            print("Detected standard checkpoint loading into AdaptedCFM. Remapping keys...")
            for k, v in cfm_state.items():
                if k.startswith('estimator.'):
                    new_key = f"base_cfm.{k}"
                    new_cfm_state[new_key] = v
                else:
                    new_cfm_state[k] = v
            state_dict['cfm'] = new_cfm_state
        else:
            print("Checkpoint seems to match AdaptedCFM structure.")

    # Load state dict manually to handle the remapping
    for key in model:
        if key in state_dict:
            try:
                model[key].load_state_dict(state_dict[key], strict=False)
                print(f"Loaded {key}")
            except Exception as e:
                print(f"Error loading {key}: {e}")
        else:
            print(f"Warning: Key {key} not found in checkpoint")

    # model, _, _, _ = load_checkpoint(
    #     model,
    #     None,
    #     dit_checkpoint_path,
    #     load_only_params=True,
    #     ignore_modules=[],
    #     is_distributed=False,
    # )

    for key in model:
        model[key].eval()
        model[key].to(device)
    
    # Setup caches
    if isinstance(model.cfm, AdaptedCFM):
        if hasattr(model.cfm.base_cfm.estimator, "setup_caches"):
            model.cfm.base_cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
    else:
        if hasattr(model.cfm.estimator, "setup_caches"):
            model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus
    campplus_ckpt_path = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    vocoder_type = model_params.vocoder.type
    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = model_params.vocoder.name
        # Note: Assuming bigvgan_cache is defined or handled by from_pretrained if needed, 
        # but copying logic from my_inference.py which had a typo/redundancy.
        # Fixing it to just use the name.
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=False)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        vocoder_fn = bigvgan_model
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")

    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )


@torch.no_grad()
def run_inference(args, model_bundle, device=torch.device("cuda")):
    dit_config_path = args.config
    config = yaml.safe_load(open(dit_config_path, "r"))

    use_style_residual = config['model_params']['length_regulator'].get('use_style_residual', False)

    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = model_bundle
    fp16 = args.fp16
    sr = mel_fn_args['sampling_rate']
    f0_condition = args.f0_condition
    forch_pitch_shift = args.semi_tone_shift

    source = args.source
    target_name = args.target
    print(f"Source: {source}, Target: {target_name}")
    
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    exp_path = os.path.join(args.output, args.expname)
    
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target_name, sr=sr)[0]

    sr_proc = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr_proc // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    time_vc_start = time.time()
    
    # Resample for semantic extraction
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    
    # Semantic extraction with chunking if needed
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_chunk = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_chunk)
            else:
                S_alt_list.append(S_chunk[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    S_ori = semantic_fn(ori_waves_16k)

    mel = mel_fn(source_audio.float())
    mel2 = mel_fn(ref_audio.float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        shifted_log_f0_alt = log_f0_alt.clone()
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)

        # automatic f0 adjust
        shifted_f0_alt, pitch_shift = preprocess_voice_conversion(
            voiced_f0_ori=voiced_F0_ori,
            voiced_f0_alt=voiced_F0_alt,
            shifted_f0_alt=shifted_f0_alt,
            enable_adaptive=True,
            max_shift_semitones=24,
            forch_pitch_shift=forch_pitch_shift,
        )
        print(f'automatic pitch shift {pitch_shift} semi tones')
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    # Length regulation
    cond, _, codes, commitment_loss, codebook_loss, style_cond = model.length_regulator(
        S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt, style=style2, return_style_residual=True
    )
    prompt_condition, _, codes, commitment_loss, codebook_loss, style_prompt = model.length_regulator(
        S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori, style=style2, return_style_residual=True
    )

    max_source_window = max_context_window - mel2.size(2)
    if max_source_window <= 0:
        print(f"Warning: Prompt length ({mel2.size(2)}) exceeds or is too close to max context window ({max_context_window}).")
        print("Reducing prompt length to fit...")
        # Keep at least 10 seconds for generation
        min_gen_frames = int(10 * sr_proc / hop_length)
        max_prompt_len = max_context_window - min_gen_frames
        if max_prompt_len < 100: # If context is too small
             raise ValueError("Context window too small for generation.")
        
        # Trim prompt features
        mel2 = mel2[:, :, :max_prompt_len]
        target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
        prompt_condition = prompt_condition[:, :max_prompt_len]
        if style_prompt is not None:
            style_prompt = style_prompt[:, :max_prompt_len]
        
        max_source_window = max_context_window - mel2.size(2)
        print(f"New prompt length: {mel2.size(2)}, Max source window: {max_source_window}")

    processed_frames = 0
    generated_wave_chunks = []
    
    print(f"Starting generation loop. Total frames to process: {cond.size(1)}")
    
    # Stream generation
    while processed_frames < cond.size(1):
        print(f"Processing chunk starting at frame {processed_frames}...")
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        
        if use_style_residual:
            chunk_style_cond = style_cond[:, processed_frames:processed_frames + max_source_window]
            cat_style_cond = torch.cat([style_prompt, chunk_style_cond], dim=1)
        else:
            cat_style_cond = None
            
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            # Voice Conversion Inference
            # Check if using AdaptedCFM to pass use_rva
            kwargs = {
                'inference_cfg_rate': inference_cfg_rate,
                'style_r': cat_style_cond
            }
            
            if isinstance(model.cfm, AdaptedCFM):
                kwargs['use_rva'] = args.use_rva
            
            vc_target = model.cfm.inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                mel2, 
                style2, 
                None, 
                diffusion_steps,
                **kwargs
            )

            vc_target = vc_target[:, :, mel2.size(-1):]
            
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        
        print(f"Chunk processed. New processed_frames: {processed_frames}")
            
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()

    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")
    os.makedirs(exp_path, exist_ok=True)
    src_name = os.path.basename(source).split(".")[0]
    tgt_name = os.path.basename(target_name).split(".")[0]
    
    if hasattr(args, "uuid") and args.uuid:
        vc_name = f'{src_name}_{tgt_name}_' + args.uuid + '.wav'
    else:
        vc_name = f"{tgt_name}_{src_name}_{pitch_shift}.wav"
        
    output_path = os.path.join(exp_path, vc_name)
    torchaudio.save(output_path, vc_wave.cpu(), sr)
    print(f"Saved output to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Path to source audio")
    parser.add_argument("--target", type=str, required=True, help="Path to target reference audio")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--expname", type=str, default="test_sfa_rva")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--fp16", type=str, default="true")
    parser.add_argument("--accompany", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/YingMusic-SVC.yml")
    
    # New arguments for SFA-LoRA and RVA
    parser.add_argument("--use-lora", action='store_true', help="Use SFA-LoRA adapter")
    parser.add_argument("--use-rva", action='store_true', help="Use Residual Velocity Adapter")
    
    args = parser.parse_args()

    args.cuda = torch.device(f"cuda:{args.cuda}")
    args.fp16 = str2bool(args.fp16)
    if args.fp16:
        print('Start fp16 to accelerate inference！')

    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    args.f0_condition = True
    args.semi_tone_shift = None    # If None, the tone is automatically sandhi

    args.output = './outputs'
    os.makedirs(args.output, exist_ok=True)

    models = load_models_api(args, device=args.cuda)
    vc = run_inference(args, models, device=args.cuda)
    
    if args.accompany:
        vc_t = vc.split('/')
        a,b = '/'.join(vc_t[:-1]), vc_t[-1]
        os.makedirs(a + '/accompany', exist_ok=True)
        op = a + '/accompany/'+b
        echo_then_reverb_save(vc,op,args.accompany)
