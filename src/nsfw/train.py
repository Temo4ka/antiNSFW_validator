import pathlib
import random
import subprocess

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from nsfw.baseline_model import CNNBaselineModel
from nsfw.conf_analyzer import create_transforms
from nsfw.data import ImageDataSet, LoadDataFrom
from nsfw.dvc_utils import ensure_data_downloaded
from nsfw.model import ConvNextModel


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
    # Загрузка данных через DVC
    data_dir = pathlib.Path(cfg.data.paths.nsfw_dir).parent
    ensure_data_downloaded(str(data_dir), use_dvc=cfg.data.get("use_dvc", True))

    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

    nsfw_images, nsfw_labels = LoadDataFrom(
        cfg.data.paths.nsfw_dir, cfg.data.paths.nsfw_label, cfg.data.paths.file_pattern
    )
    sfw_images, sfw_labels = LoadDataFrom(
        cfg.data.paths.sfw_dir, cfg.data.paths.sfw_label, cfg.data.paths.file_pattern
    )

    all_images = nsfw_images + sfw_images
    all_labels = nsfw_labels + sfw_labels

    train_data, train_labels, val_data, val_labels = shuffle_and_split_data(
        all_images,
        all_labels,
        split_ratio=cfg.data.split.ratio,
        seed=cfg.data.split.seed,
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
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers,
        shuffle=False,
    )

    config_dict = OmegaConf.to_container(cfg, resolve=True)

    if "CNN" in config_dict:
        model = CNNBaselineModel(config_dict)
    else:
        model = ConvNextModel(config_dict)

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        .decode()
        .strip()
    )

    logger = MLFlowLogger(
        experiment_name=cfg.logging.mlflow.experiment_name,
        tracking_uri=cfg.logging.mlflow.tracking_uri,
        tags={"git_commit": git_commit},
        log_model=True,
    )
    logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.trainer.checkpoint.dirpath,
        filename=cfg.trainer.checkpoint.filename,
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=cfg.trainer.checkpoint.save_top_k,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
