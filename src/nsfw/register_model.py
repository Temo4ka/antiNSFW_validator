import pathlib

import hydra
import mlflow
import mlflow.pytorch
from omegaconf import DictConfig

from nsfw.model import ConvNextModel


@hydra.main(
    version_base=None, config_path="../../configs", config_name="register_model_config"
)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    checkpoint_path = pathlib.Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {checkpoint_path}")

    print(f"Загрузка модели из {checkpoint_path}...")
    model = ConvNextModel.load_from_checkpoint(str(checkpoint_path))
    model.eval()

    # Получаем run_id из чекпоинта или используем указанный
    run_id = cfg.mlflow.get("run_id", None)

    if run_id is None:
        # Пытаемся найти run по имени чекпоинта
        print("Поиск run в MLflow...")
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=10,
            )
            # Ищем run, который содержит этот чекпоинт
            for run in runs:
                if checkpoint_path.name in str(run.data.tags.get("checkpoint", "")):
                    run_id = run.info.run_id
                    print(f"Найден run: {run_id}")
                    break

    if run_id is None:
        print(
            "Предупреждение: run_id не найден, "
            "модель будет зарегистрирована без связи с run"
        )

    with mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run():
        # Регистрируем модель
        model_name = cfg.model_name

        print(f"Регистрация модели '{model_name}' в MLflow...")
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        print(f"Модель '{model_name}' успешно зарегистрирована в MLflow Model Registry")


if __name__ == "__main__":
    main()
