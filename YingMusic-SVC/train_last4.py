import os
import sys
import glob
import yaml
import argparse
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

from modules.commons import recursive_munch, build_model, my_load_checkpoint

from data.ft_dataset import build_ft_dataloader
from hf_utils import load_custom_model_from_hf


# ============================================================================
# Fine-Grained Parameter Control Utilities
# ============================================================================

def print_model_structure(model: nn.Module, name: str = "model", max_depth: int = 3) -> None:
    """Print the hierarchical structure of a model with parameter counts."""
    print(f"\n{'='*60}")
    print(f"Model Structure: {name}")
    print(f"{'='*60}")
    
    def _print_module(module: nn.Module, prefix: str, depth: int):
        if depth > max_depth:
            return
        for child_name, child in module.named_children():
            n_params = sum(p.numel() for p in child.parameters())
            n_trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
            status = "✓" if n_trainable > 0 else "✗"
            print(f"{prefix}├─ {child_name}: {child.__class__.__name__} "
                  f"[params: {n_params:,}, trainable: {n_trainable:,}] {status}")
            _print_module(child, prefix + "│  ", depth + 1)
    
    _print_module(model, "", 0)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"{'='*60}\n")


def get_parameter_groups(model: nn.Module) -> Dict[str, List[nn.Parameter]]:
    """Get named parameter groups from a model for fine-grained control."""
    param_groups = {}
    for name, param in model.named_parameters():
        # Extract top-level module name
        parts = name.split('.')
        group_name = parts[0] if parts else 'root'
        if group_name not in param_groups:
            param_groups[group_name] = []
        param_groups[group_name].append((name, param))
    return param_groups


def freeze_all_params(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params_by_name(model: nn.Module, patterns: List[str]) -> int:
    """Unfreeze parameters matching any of the given patterns.
    
    Args:
        model: The model to modify
        patterns: List of substring patterns to match against parameter names
        
    Returns:
        Number of parameters unfrozen
    """
    count = 0
    for name, param in model.named_parameters():
        for pattern in patterns:
            if pattern in name:
                param.requires_grad = True
                count += param.numel()
                break
    return count


def setup_dit_finetuning(
    dit: nn.Module,
    n_tune_layers: int = 8,
    tune_norm: bool = True,
    tune_final_layers: bool = True,
    tune_embeddings: bool = False,
    tune_style: bool = True,
    verbose: bool = True
) -> List[nn.Parameter]:
    """Configure DiT model for fine-tuning with fine-grained control.
    
    Args:
        dit: The DiT model (cfm.estimator)
        n_tune_layers: Number of transformer layers to tune from the end
        tune_norm: Whether to tune the final norm layer
        tune_final_layers: Whether to tune final projection layers (final_mlp, wavenet, etc.)
        tune_embeddings: Whether to tune input embeddings (t_embedder, cond_projection, etc.)
        tune_style: Whether to tune style-related components
        verbose: Whether to print tuning information
        
    Returns:
        List of trainable parameters
    """
    # Freeze everything first
    freeze_all_params(dit)
    
    trainable_params = []
    tuned_components = []
    
    # 1. Tune last N transformer layers
    if hasattr(dit, 'transformer') and hasattr(dit.transformer, 'layers'):
        n_layers = len(dit.transformer.layers)
        start_idx = max(0, n_layers - n_tune_layers)
        for i in range(start_idx, n_layers):
            for param in dit.transformer.layers[i].parameters():
                param.requires_grad = True
                trainable_params.append(param)
        tuned_components.append(f"transformer.layers[{start_idx}:{n_layers}] ({n_tune_layers} layers)")
    
    # 2. Tune final norm
    if tune_norm and hasattr(dit, 'transformer') and hasattr(dit.transformer, 'norm'):
        for param in dit.transformer.norm.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        tuned_components.append("transformer.norm")
    
    # 3. Tune final projection layers
    if tune_final_layers:
        final_layer_names = ['final_mlp', 'final_layer', 'conv1', 'conv2', 'wavenet', 'res_projection']
        for layer_name in final_layer_names:
            if hasattr(dit, layer_name):
                layer = getattr(dit, layer_name)
                for param in layer.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                tuned_components.append(layer_name)
    
    # 4. Tune embeddings
    if tune_embeddings:
        embedding_names = ['t_embedder', 't_embedder2', 'cond_projection', 'cond_x_merge_linear']
        for emb_name in embedding_names:
            if hasattr(dit, emb_name):
                emb = getattr(dit, emb_name)
                for param in emb.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                tuned_components.append(emb_name)
    
    # 5. Tune style-related components
    if tune_style:
        style_names = ['style_in', 'skip_linear']
        for style_name in style_names:
            if hasattr(dit, style_name):
                style_layer = getattr(dit, style_name)
                for param in style_layer.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                tuned_components.append(style_name)
    
    if verbose:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in dit.parameters())
        print(f"\n[DiT Fine-tuning Configuration]")
        print(f"  Tuned components: {', '.join(tuned_components)}")
        print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    
    return trainable_params


def setup_length_regulator_finetuning(
    lr_module: nn.Module,
    tune_model: bool = True,
    tune_embeddings: bool = True,
    tune_style_residual: bool = True,
    tune_vq: bool = False,
    verbose: bool = True
) -> List[nn.Parameter]:
    """Configure Length Regulator for fine-tuning.
    
    Args:
        lr_module: The length regulator module
        tune_model: Whether to tune the main conv model
        tune_embeddings: Whether to tune f0 and content embeddings
        tune_style_residual: Whether to tune style residual components
        tune_vq: Whether to tune vector quantization components
        verbose: Whether to print tuning information
        
    Returns:
        List of trainable parameters
    """
    # Freeze everything first
    freeze_all_params(lr_module)
    
    trainable_params = []
    tuned_components = []
    
    # 1. Tune main model
    if tune_model and hasattr(lr_module, 'model'):
        for param in lr_module.model.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        tuned_components.append("model")
    
    # 2. Tune embeddings
    if tune_embeddings:
        emb_names = ['embedding', 'f0_embedding', 'content_in_proj', 'extra_codebooks']
        for emb_name in emb_names:
            if hasattr(lr_module, emb_name):
                emb = getattr(lr_module, emb_name)
                if isinstance(emb, nn.ModuleList):
                    for sub_emb in emb:
                        for param in sub_emb.parameters():
                            param.requires_grad = True
                            trainable_params.append(param)
                else:
                    for param in emb.parameters():
                        param.requires_grad = True
                        trainable_params.append(param)
                tuned_components.append(emb_name)
    
    # 3. Tune style residual components
    if tune_style_residual:
        style_names = ['f0_to_style_proj', 'f02style_mlp']
        for style_name in style_names:
            if hasattr(lr_module, style_name):
                style_layer = getattr(lr_module, style_name)
                for param in style_layer.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                tuned_components.append(style_name)
    
    # 4. Tune VQ components
    if tune_vq and hasattr(lr_module, 'vq'):
        for param in lr_module.vq.parameters():
            param.requires_grad = True
            trainable_params.append(param)
        tuned_components.append("vq")
    
    if verbose:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in lr_module.parameters())
        if n_total > 0:
            print(f"\n[Length Regulator Fine-tuning Configuration]")
            print(f"  Tuned components: {', '.join(tuned_components)}")
            print(f"  Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
    
    return trainable_params


class YingMusicTrainer:
    def __init__(
        self,
        config_path: str,
        dataset_dir: str,
        run_name: str,
        pretrained_ckpt: str | None = None,
        batch_size: int = 2,
        num_workers: int = 0,
        steps: int = 1000,
        max_epochs: int = 1000,
        save_every: int = 500,
        device: str = "cuda:0",
        # Fine-tuning configuration
        n_tune_layers: int = 8,
        tune_norm: bool = True,
        tune_final_layers: bool = True,
        tune_embeddings: bool = False,
        tune_style: bool = True,
        tune_lr_model: bool = False,
        tune_lr_embeddings: bool = False,
        tune_lr_style_residual: bool = True,
        print_structure: bool = True,
    ) -> None:
        self.device = device
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        self.cfg = recursive_munch(config)

        # logging dir
        self.log_dir = os.path.join("runs", run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # save a copy of config
        with open(os.path.join(self.log_dir, os.path.basename(config_path)), "w", encoding="utf-8") as wf:
            yaml.safe_dump(config, wf, sort_keys=False)

        # dataloader
        self.sr = self.cfg.preprocess_params.sr
        self.train_loader = build_ft_dataloader(
            dataset_dir,
            self.cfg.preprocess_params.spect_params,
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # build feature extractors
        self._build_semantic_fn()
        self._build_style_encoder()
        self._build_f0_extractor()

        # build model
        self.model = build_model(self.cfg.model_params, stage="DiT")
        _ = [self.model[k].to(self.device) for k in self.model]
        # some modules (like estimators) may need cache init
        if hasattr(self.model.cfm, "estimator") and hasattr(self.model.cfm.estimator, "setup_caches"):
            # heuristic cache size
            self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        # ====================================================================
        # Fine-Grained Parameter Control
        # ====================================================================
        
        # Print model structure before configuring
        if print_structure:
            print("\n" + "="*80)
            print("MODEL ARCHITECTURE ANALYSIS")
            print("="*80)
            print_model_structure(self.model.cfm.estimator, "DiT (cfm.estimator)", max_depth=2)
            print_model_structure(self.model.length_regulator, "Length Regulator", max_depth=2)
        
        # Configure DiT fine-tuning
        dit_params = setup_dit_finetuning(
            dit=self.model.cfm.estimator,
            n_tune_layers=n_tune_layers,
            tune_norm=tune_norm,
            tune_final_layers=tune_final_layers,
            tune_embeddings=tune_embeddings,
            tune_style=tune_style,
            verbose=True
        )
        
        # Configure Length Regulator fine-tuning
        lr_params = setup_length_regulator_finetuning(
            lr_module=self.model.length_regulator,
            tune_model=tune_lr_model,
            tune_embeddings=tune_lr_embeddings,
            tune_style_residual=tune_lr_style_residual,
            tune_vq=False,
            verbose=True
        )
        
        # Collect all trainable parameters
        params = [p for p in self.model.cfm.parameters() if p.requires_grad] + \
                 [p for p in self.model.length_regulator.parameters() if p.requires_grad]
        
        # Print final summary
        total_trainable = sum(p.numel() for p in params)
        total_all = sum(p.numel() for p in self.model.cfm.parameters()) + \
                    sum(p.numel() for p in self.model.length_regulator.parameters())
        print(f"\n{'='*60}")
        print(f"TRAINING CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"Total trainable parameters: {total_trainable:,} / {total_all:,} ({100*total_trainable/total_all:.2f}%)")
        print(f"DiT: tune_layers={n_tune_layers}, norm={tune_norm}, final={tune_final_layers}, emb={tune_embeddings}, style={tune_style}")
        print(f"LR:  model={tune_lr_model}, emb={tune_lr_embeddings}, style_residual={tune_lr_style_residual}")
        print(f"{'='*60}\n")

        base_lr = float(self.cfg.loss_params.get("base_lr", 1e-5))
        self.optimizer = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.98), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(steps, 1))

        # try resume or load
        self.epoch = 0
        self.iters = 0
        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            self.model, self.optimizer, self.scheduler, self.epoch, self.iters = my_load_checkpoint(
                self.model, self.optimizer, self.scheduler, pretrained_ckpt, load_only_params=True
            )
            for key in self.model:
                if hasattr(self.model[key], 'train'):
                    self.model[key].train()
            print(f"Loaded pretrained checkpoint from {pretrained_ckpt}")
        else:
            # auto-resume latest DiT checkpoint in run dir
            latest = self._find_latest_ckpt(pattern="DiT_epoch_*_step_*.pth")
            if latest:
                self.model, self.optimizer, self.scheduler, self.epoch, self.iters = my_load_checkpoint(
                    self.model, self.optimizer, self.scheduler, latest, load_only_params=False
                )
                print(f"Auto-resumed from {latest}")

        # train control
        self.max_steps = steps
        self.max_epochs = max_epochs
        self.save_every = save_every
        self.log_interval = 10
        self.use_balance_loss = bool(self.cfg.loss_params.get("use_balance_loss", True))
        self.loss_ema = 0.0
        self.loss_smooth = 0.99

    def _find_latest_ckpt(self, pattern: str) -> str | None:
        files = glob.glob(os.path.join(self.log_dir, pattern))
        if not files:
            return None
        files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return files[-1]

    def _build_style_encoder(self):
        from modules.campplus.DTDNN import CAMPPlus
        self.camp = CAMPPlus(feat_dim=80, embedding_size=192)
        # try local path first, else download from HF
        ckpt_name = getattr(self.cfg.model_params.style_encoder, "campplus_path", "campplus_cn_common.bin")
        ckpt_path = ckpt_name if os.path.exists(ckpt_name) else load_custom_model_from_hf("funasr/campplus", ckpt_name, None)
        sd = torch.load(ckpt_path, map_location="cpu")
        self.camp.load_state_dict(sd)
        self.camp.eval().to(self.device)
        self.spk_encoder = self.camp

    def _build_f0_extractor(self):
        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)

    def _build_semantic_fn(self):
        tok = self.cfg.model_params.speech_tokenizer
        typ = tok.get("type", "whisper")
        if typ == "whisper":
            from transformers import AutoFeatureExtractor, WhisperModel
            name = tok["name"]
            self.whisper = WhisperModel.from_pretrained(name).to(self.device)
            self.whisper_extractor = AutoFeatureExtractor.from_pretrained(name)
            # free decoder to save memory
            if hasattr(self.whisper, "decoder"):
                del self.whisper.decoder

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                ori_inputs = self.whisper_extractor(
                    [w16k.cpu().numpy() for w16k in waves_16k],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                feats = self.whisper._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(self.device)
                with torch.no_grad():
                    enc_out = self.whisper.encoder(
                        feats.to(self.whisper.encoder.dtype),
                        head_mask=None,
                        output_attentions=False,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                S = enc_out.last_hidden_state.to(torch.float32)
                # time align to 20ms per step (Whisper ~ 320 samples at 16k)
                return S[:, : waves_16k.size(-1) // 320 + 1]

            self.semantic_fn = semantic_fn
        elif typ == "xlsr":
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
            name = tok["name"]
            out_layer = int(tok.get("output_layer", 12))
            self.wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(name)
            self.wav2vec = Wav2Vec2Model.from_pretrained(name)
            self.wav2vec.encoder.layers = self.wav2vec.encoder.layers[:out_layer]
            self.wav2vec = self.wav2vec.eval().half().to(self.device)

            def semantic_fn(waves_16k: torch.Tensor) -> torch.Tensor:
                arr = [w.cpu().numpy() for w in waves_16k]
                inputs = self.wav2vec_extractor(
                    arr, return_tensors="pt", return_attention_mask=True, padding=True, sampling_rate=16000
                ).to(self.device)
                with torch.no_grad():
                    out = self.wav2vec(inputs.input_values.half())
                return out.last_hidden_state.float()

            self.semantic_fn = semantic_fn
        else:
            raise ValueError(f"Unsupported speech_tokenizer.type: {typ}")

    def _extract_style(self, waves_16k: torch.Tensor, wave_lengths_16k: torch.Tensor) -> torch.Tensor:
        feat_list = []
        for b in range(waves_16k.size(0)):
            feat = kaldi.fbank(
                waves_16k[b:b+1, : wave_lengths_16k[b]], num_mel_bins=80, dither=0, sample_frequency=16000
            )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.spk_encoder(feat.unsqueeze(0))
                y_list.append(y)
        return torch.cat(y_list, dim=0)

    def _train_step(self, batch):
        waves, mels, wave_lens, mel_lens = batch
        B = waves.size(0)

        # resample
        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lens_16k = (wave_lens.float() * 16000 / self.sr).long()

        # semantic tokens
        S = self.semantic_fn(waves_16k)

        # f0
        F0 = self.rmvpe.infer_from_audio_batch(waves_16k)
        F0 = F0.to(mels.device).float()

        # length regulator to align content to mel length
        cond, _, _, commit_loss, codebook_loss = self.model.length_regulator(
            S, ylens=mel_lens, f0=F0
        )
        commit_loss = commit_loss if commit_loss is not None else 0.0
        codebook_loss = codebook_loss if codebook_loss is not None else 0.0

        # prompt length per sample (random)
        prompt_len_max = mel_lens - 1
        prompt_lens = (torch.rand([B], device=mels.device) * prompt_len_max).floor().long()
        zeros_prob = torch.rand([B], device=mels.device) < 0.1
        prompt_lens = torch.where(zeros_prob, torch.zeros_like(prompt_lens), prompt_lens)

        # ensure equal time length for x1(target) and cond
        T = min(mels.size(2), cond.size(1))
        x1 = mels[:, :, :T]
        cond = cond[:, :T]
        x_lens = torch.clamp(mel_lens, max=T)

        # style embedding (per sample)
        style = self._extract_style(waves_16k, wave_lens_16k)

        loss_cfm, _ = self.model.cfm(
            x1, x_lens, prompt_lens, cond, style, balance_loss=self.use_balance_loss
        )
        loss_total = loss_cfm + 0.05 * float(commit_loss) + 0.15 * float(codebook_loss)

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss_cfm.detach().item()

    def _save_ckpt(self):
        state = {
            'net': {k: self.model[k].state_dict() for k in self.model},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'iters': self.iters,
            'epoch': self.epoch,
        }
        path = os.path.join(self.log_dir, f"DiT_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth")
        torch.save(state, path)
        # cleanup older
        ckpts = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth"))
        if len(ckpts) > 2:
            ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for p in ckpts[:-2]:
                try:
                    os.remove(p)
                except Exception:
                    pass

    def train(self):
        self.loss_ema = 0.0
        for epoch in range(self.epoch, self.max_epochs):
            self.epoch = epoch
            for batch in tqdm(self.train_loader):
                batch = [b.to(self.device) for b in batch]
                loss = self._train_step(batch)
                self.loss_ema = self.loss_ema * self.loss_smooth + loss * (1 - self.loss_smooth) if self.iters > 0 else loss
                if self.iters % self.log_interval == 0:
                    print(f"epoch {self.epoch}, step {self.iters}, loss: {self.loss_ema:.4f}")
                self.iters += 1

                if self.iters % self.save_every == 0:
                    self._save_ckpt()

                if self.iters >= self.max_steps:
                    break
            if self.iters >= self.max_steps:
                break

        # save final
        final_path = os.path.join(self.log_dir, "ft_model.pth")
        torch.save({'net': {k: self.model[k].state_dict() for k in self.model}}, final_path)
        print(f"Final model saved at {final_path}")


def main():
    parser = argparse.ArgumentParser(description="YingMusic-SVC Fine-tuning with Fine-Grained Parameter Control")
    
    # Basic training arguments
    parser.add_argument('--config', type=str, default='configs/YingMusic-SVC.yml')
    parser.add_argument('--dataset-dir', type=str, required=True)
    parser.add_argument('--run-name', type=str, default='ymsvc_run')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--gpu', type=int, default=0)
    
    # Fine-grained parameter control arguments
    parser.add_argument('--n-tune-layers', type=int, default=8,
                        help='Number of DiT transformer layers to fine-tune from the end')
    parser.add_argument('--tune-norm', action='store_true', default=True,
                        help='Fine-tune the final norm layer')
    parser.add_argument('--no-tune-norm', action='store_false', dest='tune_norm',
                        help='Do not fine-tune the final norm layer')
    parser.add_argument('--tune-final-layers', action='store_true', default=True,
                        help='Fine-tune final projection layers (final_mlp, wavenet, etc.)')
    parser.add_argument('--no-tune-final-layers', action='store_false', dest='tune_final_layers',
                        help='Do not fine-tune final projection layers')
    parser.add_argument('--tune-embeddings', action='store_true', default=False,
                        help='Fine-tune input embeddings (t_embedder, cond_projection)')
    parser.add_argument('--tune-style', action='store_true', default=True,
                        help='Fine-tune style-related components')
    parser.add_argument('--no-tune-style', action='store_false', dest='tune_style',
                        help='Do not fine-tune style-related components')
    parser.add_argument('--tune-lr-model', action='store_true', default=False,
                        help='Fine-tune length regulator main model')
    parser.add_argument('--tune-lr-embeddings', action='store_true', default=False,
                        help='Fine-tune length regulator embeddings')
    parser.add_argument('--tune-lr-style-residual', action='store_true', default=True,
                        help='Fine-tune length regulator style residual components')
    parser.add_argument('--no-tune-lr-style-residual', action='store_false', dest='tune_lr_style_residual',
                        help='Do not fine-tune length regulator style residual')
    parser.add_argument('--no-print-structure', action='store_false', dest='print_structure',
                        help='Do not print model structure')
    
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    trainer = YingMusicTrainer(
        config_path=args.config,
        dataset_dir=args.dataset_dir,
        run_name=args.run_name,
        pretrained_ckpt=args.pretrained_ckpt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_every=args.save_every,
        device=device,
        # Fine-tuning configuration
        n_tune_layers=args.n_tune_layers,
        tune_norm=args.tune_norm,
        tune_final_layers=args.tune_final_layers,
        tune_embeddings=args.tune_embeddings,
        tune_style=args.tune_style,
        tune_lr_model=args.tune_lr_model,
        tune_lr_embeddings=args.tune_lr_embeddings,
        tune_lr_style_residual=args.tune_lr_style_residual,
        print_structure=getattr(args, 'print_structure', True),
    )
    trainer.train()


if __name__ == '__main__':
    if sys.platform == 'win32':
        mp.freeze_support()
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    main()
