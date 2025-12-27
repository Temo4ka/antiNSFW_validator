#!/bin/bash
# Скрипт для запуска обучения обеих моделей

cd "$(dirname "$0")"

echo "=========================================="
echo "Обучение ConvNext модели"
echo "=========================================="
poetry run python -m nsfw.train training.max_epochs=10 training.batch_size=32

echo ""
echo "=========================================="
echo "Обучение CNN Baseline модели"
echo "=========================================="
poetry run python -m nsfw.train --config-name=config_baseline training.max_epochs=10 training.batch_size=32

echo ""
echo "=========================================="
echo "Обучение завершено!"
echo "Результаты сохранены в:"
echo "  - MLflow: .mlruns/"
echo "  - Checkpoints: checkpoints/"
echo "=========================================="
