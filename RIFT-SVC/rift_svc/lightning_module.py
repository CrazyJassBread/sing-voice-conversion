import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import wandb
from functools import partial
import inspect

from pytorch_lightning import LightningModule

from rift_svc.metrics import mcd, psnr, si_snr
from rift_svc.feature_extractors import get_mel_spectrogram
from rift_svc.nsf_hifigan import NsfHifiGAN
from rift_svc.utils import draw_mel_specs, l2_grad_norm


class RIFTSVCLightningModule(LightningModule):
    def __init__(
        self,
        model,
        optimizer,
        cfg,
        lr_scheduler=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.cfg = cfg
        self.eval_sample_steps = cfg['training']['eval_sample_steps']
        self.model.sample = partial(
            self.model.sample,
            steps=self.eval_sample_steps,
        )
        self.log_media_per_steps = cfg['training']['log_media_per_steps']
        self.drop_spk_prob = cfg['training']['drop_spk_prob']

        self.vocoder = None
        self.save_hyperparameters(ignore=['model', 'optimizer', 'vocoder'])

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return self.optimizer
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
            }
        }

    def training_step(self, batch, batch_idx):
        mel = batch['mel']
        spk_id = batch['spk_id']
        f0 = batch['f0']
        rms = batch['rms']
        cvec = batch['cvec']
        frame_len = batch['frame_len']

        drop_speaker = False
        if self.drop_spk_prob > 0:
            batch_size = spk_id.shape[0]
            num_drop = int(batch_size * self.drop_spk_prob)
            drop_speaker = torch.zeros(batch_size, dtype=torch.bool, device=spk_id.device)
            drop_speaker[:num_drop] = True
            # Randomly shuffle the drop mask
            drop_speaker = drop_speaker[torch.randperm(batch_size)]

        loss, _ = self.model(
            mel,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            drop_speaker=drop_speaker,
            frame_len=frame_len,
        )

        # Log metrics - compatible with both loggers
        self._log_scalar("train/loss", loss.item(), prog_bar=True)
        
        return loss
    
    def on_validation_start(self):
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()
        if not self.trainer.is_global_zero:
            return

        if self.vocoder is None:
            self.vocoder =  NsfHifiGAN(
                'pretrained/nsf_hifigan_44.1k_hop512_128bin_2024.02/model.ckpt').to(self.device)
        else:
            self.vocoder = self.vocoder.to(self.device)
        
        self.mcd = []
        self.si_snr = []
        self.psnr = []
        self.mse = []


    def on_validation_end(self, log=True):
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.train()
        if not self.trainer.is_global_zero:
            return

        if self.vocoder is not None:
            self.vocoder = self.vocoder.cpu()
            gc.collect()
            torch.cuda.empty_cache()
        
        metrics = {
            'val/mcd': np.mean(self.mcd),
            'val/si_snr': np.mean(self.si_snr),
            'val/psnr': np.mean(self.psnr),
            'val/mse': np.mean(self.mse)
        }

        if log:
            # Log metrics - compatible with both loggers
            for metric_name, metric_value in metrics.items():
                self._log_scalar(metric_name, metric_value)


    def validation_step(self, batch, batch_idx, log=True):
        """
        Process validation step and log metrics and media.
        
        Args:
            batch: Input batch
            batch_idx: Batch index
            log: Whether to log or not
        """
        # Skip if not the main process or logging is disabled
        if not self.trainer.is_global_zero:
            return
        
        # Get step and interval info
        global_step = self.global_step
        log_media_every_n_steps = self.log_media_every_n_steps
        
        # Extract input data
        spk_id = batch['spk_id']
        mel_gt = batch['mel']
        rms = batch['rms']
        f0 = batch['f0']
        cvec = batch['cvec']
        frame_len = batch['frame_len']
        cvec_ds = batch.get('cvec_ds', None)

        # Generate output
        mel_gen, _ = self.model.sample(
            src_mel=mel_gt,
            spk_id=spk_id,
            f0=f0,
            rms=rms,
            cvec=cvec,
            frame_len=frame_len,
            bad_cvec=cvec_ds,
        )
        mel_gen = mel_gen.float()
        mel_gt = mel_gt.float()

        # Process each sample in the batch
        for i in range(mel_gen.shape[0]):
            sample_idx = batch_idx * mel_gen.shape[0] + i
            
            # Generate audio using vocoder
            wav_gen = self.vocoder(mel_gen[i:i+1, :frame_len[i], :].transpose(1, 2), f0[i:i+1, :frame_len[i]])
            wav_gt = self.vocoder(mel_gt[i:i+1, :frame_len[i], :].transpose(1, 2), f0[i:i+1, :frame_len[i]])
            wav_gen = wav_gen.squeeze(0)
            wav_gt = wav_gt.squeeze(0)

            # Generate mel spectrograms
            mel_gen_i = get_mel_spectrogram(wav_gen).transpose(1, 2)
            mel_gt_i = get_mel_spectrogram(wav_gt).transpose(1, 2)

            # Clip values to valid range
            mel_min, mel_max = self.model.mel_min, self.model.mel_max
            mel_gen_i = torch.clip(mel_gen_i, min=mel_min, max=mel_max)
            mel_gt_i = torch.clip(mel_gt_i, min=mel_min, max=mel_max)

            # Calculate metrics
            self.mcd.append(mcd(mel_gen_i, mel_gt_i).cpu().item())
            self.si_snr.append(si_snr(mel_gen_i, mel_gt_i).cpu().item())
            self.psnr.append(psnr(mel_gen_i, mel_gt_i).cpu().item())
            self.mse.append(F.mse_loss(mel_gen_i, mel_gt_i).cpu().item())

            if log:
                # Create cache directory if it doesn't exist
                os.makedirs('.cache', exist_ok=True)
                
                # Log generated audio at specified intervals
                if global_step % log_media_every_n_steps == 0:
                    audio_path = f".cache/spk-{spk_id[i].item()}_{sample_idx}_gen.wav"
                    torchaudio.save(audio_path, wav_gen.cpu().to(torch.float32), 44100)
                    self._log_audio(self.logger, f"val-audio/spk-{spk_id[i].item()}_{sample_idx}-gen", audio_path, global_step)
                
                # Log ground truth audio only at the first step
                if global_step == 0:
                    gt_audio_path = f".cache/spk-{spk_id[i].item()}_{sample_idx}_gt.wav"
                    torchaudio.save(gt_audio_path, wav_gt.cpu().to(torch.float32), 44100)
                    self._log_audio(self.logger, f"val-audio/spk-{spk_id[i].item()}_{sample_idx}-gt", gt_audio_path, global_step)

                # Log mel spectrograms at specified intervals
                if global_step % log_media_every_n_steps == 0:
                    # Create mel spectrogram visualization
                    data_gt = mel_gt_i.squeeze().T.cpu().numpy()
                    data_gen = mel_gen_i.squeeze().T.cpu().numpy()
                    data_abs_diff = data_gen - data_gt
                    cache_path = f".cache/{sample_idx}_mel.jpg"
                    draw_mel_specs(data_gt, data_gen, data_abs_diff, cache_path)
                    self._log_image(self.logger, f"val-mel/{sample_idx}_mel", cache_path, global_step)
    
    def on_test_start(self):
        self.on_validation_start()
    
    def on_test_end(self):
        self.on_validation_end(log=False)

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx, log=False)

    def on_before_optimizer_step(self, optimizer):
        # Calculate gradient norm
        norm = l2_grad_norm(self.model)

        # Log gradient norm
        self._log_scalar("train/grad_norm", norm)

    @property
    def global_step(self):
        return self.trainer.global_step

    @property
    def log_media_every_n_steps(self):
        if self.log_media_per_steps is not None:
            return self.log_media_per_steps
        if self.save_every_n_steps is None:
            return self.trainer.val_check_interval
        return self.save_every_n_steps
    
    @property
    def save_every_n_steps(self):
        for callback in self.trainer.callbacks:
            if hasattr(callback, '_every_n_train_steps'):
                return callback._every_n_train_steps
        return None
    
    @property
    def is_using_wandb(self):
        """
        Check if WandB logger is being used.
        
        Returns:
            bool: True if WandB logger is being used, False otherwise
        """
        from pytorch_lightning.loggers import WandbLogger
        if isinstance(self.logger, WandbLogger):
            return True
        return False
    
    @property
    def is_using_tensorboard(self):
        """
        Check if TensorBoard logger is being used.
        
        Returns:
            bool: True if TensorBoard logger is being used, False otherwise
        """
        from pytorch_lightning.loggers import TensorBoardLogger
        if isinstance(self.logger, TensorBoardLogger):
            return True
        return False
    
    @property
    def logger_type(self):
        """
        Get a string representation of the logger type.
        
        Returns:
            str: 'wandb', 'tensorboard', or 'unknown'
        """
        if self.is_using_wandb:
            return 'wandb'
        elif self.is_using_tensorboard:
            return 'tensorboard'
        else:
            return 'unknown'

    def state_dict(self, *args, **kwargs):
        # Temporarily store vocoder
        vocoder = self.vocoder
        self.vocoder = None
        
        # Get state dict without vocoder
        state = super().state_dict(*args, **kwargs)
        
        # Restore vocoder
        self.vocoder = vocoder
        return state

    # Add helper methods for logging with different logger types
    def _log_scalar(self, name, value, step=None, **kwargs):
        """
        Log a scalar value to the appropriate logger.
        
        Args:
            name: Name of the metric
            value: Value of the metric
            step: Step value (defaults to current global step if None)
            **kwargs: Additional arguments to pass to the logger
        """
        if step is None:
            step = self.global_step
        
        # Special handling for on_validation_end or on_test_end
        # Get the caller function name to determine if we're in on_validation_end
        caller_frame = inspect.currentframe().f_back
        caller_function = caller_frame.f_code.co_name
        
        if caller_function in ['on_validation_end', 'on_test_end']:
            # Use logger.experiment directly as self.log() is not allowed in these hooks
            if self.is_using_wandb:
                self.logger.experiment.log({name: value}, step=step)
            elif self.is_using_tensorboard:
                self.logger.experiment.add_scalar(name, value, step)
            # Add other logger types here if needed
        else:
            # Use PyTorch Lightning's built-in logging system for scalars
            # This handles different logger types automatically
            self.log(name, value, **kwargs)
        
    def _log_audio(self, logger, name, file_path, step):
        """
        Log audio to the appropriate logger.
        
        Args:
            logger: The logger instance
            name: Name of the audio
            file_path: Path to the audio file
            step: Step value
        """
        try:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                # WandbLogger
                import wandb
                logger.experiment.log({
                    name: wandb.Audio(file_path, sample_rate=44100)
                }, step=step)
            elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_audio'):
                # TensorBoardLogger
                import soundfile as sf
                audio, sample_rate = sf.read(file_path)
                logger.experiment.add_audio(name, audio, step, sample_rate=44100)
        except Exception as e:
            print(f"Warning: Failed to log audio {name}: {e}")

    def _log_image(self, logger, name, file_path, step):
        """
        Log an image to the appropriate logger.
        
        Args:
            logger: The logger instance
            name: Name of the image
            file_path: Path to the image file
            step: Step value
        """
        try:
            if hasattr(logger, 'experiment') and hasattr(logger.experiment, 'log'):
                # WandbLogger
                import wandb
                logger.experiment.log({
                    name: wandb.Image(file_path)
                }, step=step)
            elif hasattr(logger, 'experiment') and hasattr(logger.experiment, 'add_image'):
                # TensorBoardLogger
                import PIL.Image
                import numpy as np
                import torch
                image = PIL.Image.open(file_path)
                image_array = np.array(image)
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC to CHW
                logger.experiment.add_image(name, image_tensor, step)
        except Exception as e:
            print(f"Warning: Failed to log image {name}: {e}")