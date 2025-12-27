import pathlib

import hydra
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig


def load_metrics_from_mlflow(experiment_name="nsfw_detection", run_id=None):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Эксперимент '{experiment_name}' не найден")
        return None

    # Получаем последний run или указанный
    if run_id is None:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            print("Не найдено запусков в эксперименте")
            return None
        run = runs[0]
        run_id = run.info.run_id
    else:
        run = client.get_run(run_id)

    print(f"Используется run: {run_id}")

    metrics = {}
    for key in run.data.metrics.keys():
        metric_history = client.get_metric_history(run_id, key)
        metrics[key] = [m.value for m in metric_history]
        metrics[f"{key}_step"] = [m.step for m in metric_history]

    return metrics, run_id


def create_plots(metrics, output_dir="plots"):
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Train и Val Loss
    if "train_loss" in metrics and "val_loss" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(
            metrics.get("train_loss_step", range(len(metrics["train_loss"]))),
            metrics["train_loss"],
            label="Train Loss",
            marker="o",
            markersize=3,
        )
        plt.plot(
            metrics.get("val_loss_step", range(len(metrics["val_loss"]))),
            metrics["val_loss"],
            label="Val Loss",
            marker="s",
            markersize=3,
        )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path / "loss_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Сохранен график: {output_path / 'loss_curves.png'}")

    # Train и Val AUROC
    if "train_auroc" in metrics and "val_auroc" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(
            metrics.get("train_auroc_step", range(len(metrics["train_auroc"]))),
            metrics["train_auroc"],
            label="Train AUROC",
            marker="o",
            markersize=3,
        )
        plt.plot(
            metrics.get("val_auroc_step", range(len(metrics["val_auroc"]))),
            metrics["val_auroc"],
            label="Val AUROC",
            marker="s",
            markersize=3,
        )
        plt.xlabel("Step")
        plt.ylabel("AUROC")
        plt.title("Training and Validation AUROC")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.1])
        plt.tight_layout()
        plt.savefig(output_path / "auroc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Сохранен график: {output_path / 'auroc_curves.png'}")

    # Loss и AUROC
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    if "train_loss" in metrics and "val_loss" in metrics:
        ax1.plot(
            metrics.get("train_loss_step", range(len(metrics["train_loss"]))),
            metrics["train_loss"],
            label="Train Loss",
            marker="o",
            markersize=3,
        )
        ax1.plot(
            metrics.get("val_loss_step", range(len(metrics["val_loss"]))),
            metrics["val_loss"],
            label="Val Loss",
            marker="s",
            markersize=3,
        )
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Loss Curves")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Train AUROC и Val AUROC
    if "train_auroc" in metrics and "val_auroc" in metrics:
        ax2.plot(
            metrics.get("train_auroc_step", range(len(metrics["train_auroc"]))),
            metrics["train_auroc"],
            label="Train AUROC",
            marker="o",
            markersize=3,
        )
        ax2.plot(
            metrics.get("val_auroc_step", range(len(metrics["val_auroc"]))),
            metrics["val_auroc"],
            label="Val AUROC",
            marker="s",
            markersize=3,
        )
        ax2.set_xlabel("Step")
        ax2.set_ylabel("AUROC")
        ax2.set_title("AUROC Curves")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(output_path / "combined_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Сохранен график: {output_path / 'combined_metrics.png'}")


@hydra.main(
    version_base=None, config_path="../../configs/plots", config_name="plots_config"
)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    run_id = cfg.mlflow.run_id if cfg.mlflow.run_id is not None else None

    result = load_metrics_from_mlflow(cfg.mlflow.experiment_name, run_id)
    if result is None:
        return

    metrics, run_id = result

    create_plots(metrics, cfg.plots.output_dir)

    print(f"\nГрафики успешно созданы в директории: {cfg.plots.output_dir}")


if __name__ == "__main__":
    main()
