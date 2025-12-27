import json
import pathlib

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nsfw.conf_analyzer import create_transforms
from nsfw.data import ImageDataSet, LoadDataFrom
from nsfw.model import ConvNextModel


@hydra.main(
    version_base=None, config_path="../../configs/infer", config_name="infer_config"
)
def main(cfg: DictConfig):
    pattern = cfg.data.paths.file_pattern

    model = ConvNextModel.load_from_checkpoint(cfg.checkpoint)
    model.eval()

    images, _labels = LoadDataFrom(cfg.input_dir, label=0.0, type=pattern)

    if len(images) == 0:
        print(f"Warning: No images found in {cfg.input_dir} with pattern {pattern}")
        return

    val_transforms = create_transforms(cfg.data.val_transforms)
    dummy_labels = [0.0] * len(images)
    dataset = ImageDataSet(images, dummy_labels, val_transforms)

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        return images

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=collate_fn,
    )

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)

    predictions = trainer.predict(model, dataloader)

    all_probs = []
    all_preds = []

    for batch_result in predictions:
        probs = batch_result["probs"].cpu().numpy()
        preds = batch_result["preds"].cpu().numpy()
        all_probs.extend(probs.flatten())
        all_preds.extend(preds.flatten())

    results = pd.DataFrame(
        {
            "image_path": images,
            "probability": all_probs,
            "prediction": all_preds.astype(int),
            "class_name": ["NSFW" if p > cfg.threshold else "SFW" for p in all_probs],
        }
    )

    output = pathlib.Path(cfg.output)
    if output.suffix == ".json":
        results_dict = results.to_dict("records")
        with open(output, "w") as f:
            json.dump(results_dict, f, indent=2)
    else:
        results.to_csv(output, index=False)

    print(f"Predictions saved to {output}")
    print(f"Total images processed: {len(results)}")
    print(f"NSFW detected: {results['prediction'].sum()}")
    print(f"SFW detected: {len(results) - results['prediction'].sum()}")


if __name__ == "__main__":
    main()
