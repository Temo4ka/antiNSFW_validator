import pathlib
import subprocess

import hydra
import mlflow
import mlflow.pyfunc
from omegaconf import DictConfig

from nsfw.mlflow_model import NSFWModelWrapper


@hydra.main(
    version_base=None,
    config_path="../../configs/serve",
    config_name="serve_model_config",
)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    checkpoint_path = pathlib.Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        print(f"Чекпоинт не найден: {checkpoint_path}")
        return

    model_name = cfg.model_name

    model_wrapper = NSFWModelWrapper(model_path=str(checkpoint_path))

    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model_wrapper,
            registered_model_name=model_name,
            artifacts={"model": str(checkpoint_path)},
        )

    print(f"\nЗапуск MLflow Serving сервера для модели '{model_name}'...")
    print(f"Сервер доступен по адресу: " f"{cfg.serving.host}:{cfg.serving.port}")

    model_uri = f"models:/{model_name}/latest"
    cmd = [
        "mlflow",
        "models",
        "serve",
        "-m",
        model_uri,
        "--host",
        cfg.serving.host,
        "--port",
        str(cfg.serving.port),
        "--no-conda",
    ]

    print(f"Выполняется команда: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
