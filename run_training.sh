#!/bin/bash
# Скрипт для запуска обучения обеих моделей

echo "=========================================="
poetry run python -m nsfw.train training.max_epochs=10 training.batch_size=32

echo "=========================================="
poetry run python -m nsfw.train --config-name=config_baseline training.max_epochs=10 training.batch_size=32

echo "=========================================="
