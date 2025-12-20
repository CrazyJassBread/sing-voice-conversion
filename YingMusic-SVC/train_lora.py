import os
import sys
import glob
import yaml
import argparse

import torch
import torch.multiprocessing as mp
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

from modules.commons import recursive_munch, build_model, my_load_checkpoint

from data.ft_dataset import build_ft_dataloader
from hf_utils import load_custom_model_from_hf
from modules.adapters import AdaptedCFM


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
        use_lora: bool = True,
        use_rva: bool = True,
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
        
        # Inject Adapters (SFA-LoRA and RVA)
        self.use_lora = use_lora
        self.use_rva = use_rva
        
        if use_lora or use_rva:
            lora_config = {}
            if use_lora:
                # Read from config with fallback defaults
                sfa_cfg = getattr(self.cfg.model_params, 'sfa_lora', {})
                lora_config = {
                    'rank': getattr(sfa_cfg, 'rank', 8),
                    'alpha': getattr(sfa_cfg, 'alpha', 16.0),
                    'num_freq_bins': self.cfg.preprocess_params.spect_params.n_mels,
                    'dropout': getattr(sfa_cfg, 'dropout', 0.1),
                    'use_freq_gate': True,
                    'target_modules': tuple(getattr(sfa_cfg, 'target_modules', ['wqkv', 'wo'])),
                    'sr': self.sr,
                    'gate_mode': getattr(sfa_cfg, 'gate_mode', 'feature_wise'),
                }
            
            rva_config = {}
            if use_rva:
                # Read from config with fallback defaults
                rva_cfg = getattr(self.cfg.model_params, 'rva', {})
                rva_config = {
                    'in_channels': self.cfg.model_params.DiT.in_channels,
                    'hidden_channels': getattr(rva_cfg, 'hidden_channels', 256),
                    'style_dim': self.cfg.model_params.style_encoder.dim,
                    'cond_dim': self.cfg.model_params.length_regulator.channels,
                    'use_unet': getattr(rva_cfg, 'use_unet', True),
                    'guidance_schedule': getattr(rva_cfg, 'guidance_schedule', 'cosine'),
                    'alpha_min': getattr(rva_cfg, 'alpha_min', 0.0),
                    'alpha_max': getattr(rva_cfg, 'alpha_max', 0.5),
                }
            
            training_mode = 'both'
            if use_lora and not use_rva: training_mode = 'lora_only'
            if use_rva and not use_lora: training_mode = 'rva_only'
            
            print(f"Initializing AdaptedCFM with mode: {training_mode}")
            self.model.cfm = AdaptedCFM(
                self.model.cfm,
                lora_config=lora_config,
                rva_config=rva_config,
                training_mode=training_mode,
                freeze_base=True
            ).to(self.device)
            
            # some modules (like estimators) may need cache init
            # Access estimator through base_cfm
            if hasattr(self.model.cfm.base_cfm, "estimator") and hasattr(self.model.cfm.base_cfm.estimator, "setup_caches"):
                self.model.cfm.base_cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
                
            # optimizer params from adapter
            params = self.model.cfm.get_trainable_params() + list(self.model.length_regulator.parameters())
        else:
            # some modules (like estimators) may need cache init
            if hasattr(self.model.cfm, "estimator") and hasattr(self.model.cfm.estimator, "setup_caches"):
                # heuristic cache size
                self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

            # optimizer & scheduler (simple AdamW + cosine warmup)
            params = list(self.model.cfm.parameters()) + list(self.model.length_regulator.parameters())
            
        base_lr = float(self.cfg.loss_params.get("base_lr", 1e-5))
        self.optimizer = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.98), weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(steps, 1))

        # try resume or load
        self.epoch = 0
        self.iters = 0
        
        # Check if model is adapted
        is_adapted = isinstance(self.model.cfm, torch.nn.Module) and \
                     self.model.cfm.__class__.__name__ == 'AdaptedCFM'

        if pretrained_ckpt and os.path.exists(pretrained_ckpt):
            if is_adapted:
                print(f"Loading pretrained checkpoint {pretrained_ckpt} into AdaptedCFM...")
                state = torch.load(pretrained_ckpt, map_location="cpu")
                net_state = state.get("net", {})
                
                # Handle CFM keys remapping
                if 'cfm' in net_state:
                    cfm_state = net_state['cfm']
                    new_cfm_state = {}
                    # Check if checkpoint is from base model (keys start with 'estimator.')
                    has_base_prefix = any(k.startswith('base_cfm.') for k in cfm_state.keys())
                    
                    if not has_base_prefix:
                        print("Remapping checkpoint keys to AdaptedCFM structure...")
                        for k, v in cfm_state.items():
                            if k.startswith('estimator.'):
                                new_cfm_state[f'base_cfm.{k}'] = v
                            else:
                                new_cfm_state[k] = v
                        net_state['cfm'] = new_cfm_state
                
                # Load manually
                for key in self.model:
                    if key in net_state:
                        # strict=False to ignore missing adapter keys (lora/rva)
                        self.model[key].load_state_dict(net_state[key], strict=False)
                        print(f"Loaded {key}")
                
                print(f"Loaded pretrained checkpoint from {pretrained_ckpt}")
                # Reset optimizer/scheduler for fine-tuning
            else:
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

        # AdaptedCFM handles loss calculation
        if isinstance(self.model.cfm, torch.nn.Module) and self.model.cfm.__class__.__name__ == 'AdaptedCFM':
             loss_cfm, _ = self.model.cfm(
                x1, x_lens, prompt_lens, cond, style, use_rva=self.use_rva
            )
        else:
            loss_cfm, _ = self.model.cfm(
                x1, x_lens, prompt_lens, cond, style, balance_loss=self.use_balance_loss
            )
        loss_total = loss_cfm + 0.05 * float(commit_loss) + 0.15 * float(codebook_loss)

        self.optimizer.zero_grad()
        loss_total.backward()
        
        # Clip grads - handle AdaptedCFM structure
        if isinstance(self.model.cfm, torch.nn.Module) and self.model.cfm.__class__.__name__ == 'AdaptedCFM':
            # Clip trainable params
            torch.nn.utils.clip_grad_norm_(self.model.cfm.get_trainable_params(), 10.0)
        else:
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
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--use-lora', action='store_true', help='Use SFA-LoRA')
    parser.add_argument('--use-rva', action='store_true', help='Use Residual Velocity Adapter')
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
        use_lora=args.use_lora,
        use_rva=args.use_rva,
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
