import pathlib

import hydra
import torch
from omegaconf import DictConfig

from nsfw.model import ConvNextModel


@hydra.main(
    version_base=None, config_path="../../configs", config_name="export_onnx_config"
)
def main(cfg: DictConfig):
    checkpoint_path = pathlib.Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    output_path = pathlib.Path(cfg.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка модели из {checkpoint_path}...")
    model = ConvNextModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    dummy_input = torch.randn(1, 3, cfg.input_size[0], cfg.input_size[1])

    print(f"Экспорт модели в ONNX формате: {output_path}...")

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=cfg.opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=(
            {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }
            if cfg.dynamic_batch
            else None
        ),
    )

    print(f"Модель успешно экспортирована в {output_path}")
    print(f"Размер файла: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
