import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import torch
import random

from nsfw.data import ImageDataSet, LoadDataFrom
from nsfw.model import ConvNextModel
from nsfw.transforms_factory import create_transforms


def shuffle_and_split_data(data, labels, split_ratio=0.8, seed=42):
    random.seed(seed)
    combined = list(zip(data, labels))
    random.shuffle(combined)
    data_shuffled, labels_shuffled = zip(*combined)
    
    split_idx = int(len(data_shuffled) * split_ratio)
    train_data = list(data_shuffled[:split_idx])
    train_labels = list(labels_shuffled[:split_idx])
    val_data = list(data_shuffled[split_idx:])
    val_labels = list(labels_shuffled[split_idx:])
    
    return train_data, train_labels, val_data, val_labels


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    if 'seed' in cfg:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
    
    nsfw_images, nsfw_labels = LoadDataFrom(
        cfg.data.paths.nsfw_dir,
        cfg.data.paths.nsfw_label,
        cfg.data.paths.file_pattern
    )
    sfw_images, sfw_labels = LoadDataFrom(
        cfg.data.paths.sfw_dir,
        cfg.data.paths.sfw_label,
        cfg.data.paths.file_pattern
    )
    
    all_images = nsfw_images + sfw_images
    all_labels = nsfw_labels + sfw_labels
    
    train_data, train_labels, val_data, val_labels = shuffle_and_split_data(
        all_images,
        all_labels,
        split_ratio=cfg.data.split.ratio,
        seed=cfg.data.split.seed
    )
    
    # Создание transforms из конфига
    train_transforms = create_transforms(cfg.data.train_transforms)
    val_transforms = create_transforms(cfg.data.val_transforms)
    
    # Создание датасетов
    train_dataset = ImageDataSet(train_data, train_labels, train_transforms)
    val_dataset = ImageDataSet(val_data, val_labels, val_transforms)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False
    )
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    model = ConvNextModel(config_dict)
    
    if cfg.logging.logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=cfg.logging.tensorboard.save_dir,
            name=cfg.logging.tensorboard.name
        )
    else:
        logger = None
    
    # Создание trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
    )
    
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
