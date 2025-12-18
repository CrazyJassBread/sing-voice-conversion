import os
import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader

from rift_svc import DiT, RF
from rift_svc.dataset import SVCDataset, collate_fn
from rift_svc.lightning_module import RIFTSVCLightningModule
from rift_svc.utils import CustomProgressBar, load_state_dict
from rift_svc.optim import get_optimizer

torch.set_float32_matmul_precision('high')


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    train_dataset = SVCDataset(
        **cfg.dataset,
        split="train"
    )
    
    val_dataset = SVCDataset(
        **cfg.dataset,
        split="test"
    )

    transformer = DiT(
        **cfg.model,
        num_speaker=train_dataset.num_speakers,
    )

    rf = RF(
        transformer=transformer,
        time_schedule=cfg.training.time_schedule,
    )

    # Load pretrained weights if specified
    if cfg.training.get('pretrained_path', None) is not None:
        state_dict = torch.load(cfg.training.pretrained_path, map_location='cpu')
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        # Load only model weights, allowing mismatched keys for speaker embeddings
        missing_keys, unexpected_keys = load_state_dict(rf, state_dict)
        print(f"Loaded pretrained model from {cfg.training.pretrained_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
    
    if cfg.training.get('lora_training', False):
        rf.transformer.apply_lora(cfg.training.lora_rank, cfg.training.lora_alpha)
    
    if cfg.training.get('freeze_adaln_and_tembed', False):
        rf.transformer.freeze_adaln_and_tembed()

    warmup_steps = int(cfg.training.max_steps * cfg.training.warmup_ratio)
    optimizer, lr_scheduler = get_optimizer(
        cfg.training.optimizer_type,
        rf, 
        cfg.training.learning_rate, 
        eval(cfg.training.betas), 
        cfg.training.weight_decay, 
        warmup_steps,
        max_steps=cfg.training.max_steps,
        min_lr=cfg.training.get('min_lr', 0.0),
        lora_training=cfg.training.get('lora_training', False),
    )
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict['spk2idx'] = train_dataset.spk2idx
    model = RIFTSVCLightningModule(
        model=rf,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        cfg=cfg_dict
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', cfg.training.run_name),
        filename='model-{step}',
        save_top_k=-1,
        save_last='link',
        every_n_train_steps=cfg.training.save_per_steps,
        save_weights_only=cfg.training.save_weights_only,
    )

    # Logger selection based on config
    logger_type = cfg.training.get('logger', 'wandb').lower()
    run_name = cfg.training.run_name
    # Update checkpoint directory to use run_name
    checkpoint_callback.dirpath = os.path.join('ckpts', run_name)
    
    if logger_type == 'wandb':
        # Use Weights & Biases logger
        logger = WandbLogger(
            project=cfg.training.wandb_project,
            name=run_name,
            id=cfg.training.get('wandb_resume_id', None),
            resume='allow',
        )
        if logger.experiment.config:
            # Merge with existing config, giving priority to existing values
            logger.experiment.config.update(cfg_dict, allow_val_change=True)
        else:
            # If no existing config, set it directly
            logger.experiment.config.update(cfg_dict)
    elif logger_type == 'tensorboard':
        # Use TensorBoard logger
        tensorboard_log_dir = os.path.join('logs', run_name)
        logger = TensorBoardLogger(
            save_dir=tensorboard_log_dir,
            name=None,  # Use the directory as is without adding another subfolder
            version='',  # Don't add version subdirectory
        )
    else:
        raise ValueError(f"Invalid logger type: {logger_type}")

    callbacks = [checkpoint_callback, CustomProgressBar()]
    if lr_scheduler is not None:
        callbacks.append(LearningRateMonitor(logging_interval='step'))

    trainer = pl.Trainer(
        max_steps=cfg.training.max_steps,
        accelerator='gpu',
        devices='auto',
        strategy='auto',
        precision='bf16-mixed',
        accumulate_grad_batches=cfg.training.grad_accumulation_steps,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=cfg.training.test_per_steps,
        check_val_every_n_epoch=None,
        gradient_clip_val=cfg.training.max_grad_norm,
        gradient_clip_algorithm='norm',
        log_every_n_steps=1,
    )

    if hasattr(optimizer, 'train'):
        optimizer.train()

    trainer.fit(
        model,
        train_dataloaders=DataLoader(
            train_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            persistent_workers=True,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        ),
        val_dataloaders=DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size_per_gpu,
            num_workers=cfg.training.num_workers,
            collate_fn=collate_fn,
        ),
        ckpt_path=cfg.training.get('resume_from_checkpoint', None),
    )

if __name__ == "__main__":
    main()