# NSFW Image Classification

Проект для классификации изображений на NSFW (Not Safe For Work) и SFW (Safe For Work) контент с использованием глубокого обучения. Модель предназначена для автоматического определения неподобающего контента в изображениях, что может быть полезно для модерации аватарок, постов в социальных сетях и мессенджерах.

## Содержание

- [Постановка задачи](#постановка-задачи)
- [Формат данных](#формат-данных)
- [Метрики](#метрики)
- [Валидация и тест](#валидация-и-тест)
- [Датасеты](#датасеты)
- [Моделирование](#моделирование)
- [Структура проекта](#структура-проекта)
- [Setup](#setup)
- [Train](#train)
- [Production Preparation](#production-preparation)
- [Infer](#infer)

## Постановка задачи

Основная задача проекта - обучить модель, которая сможет определять наличие неподобающего контента на фотографии. Это полезно для:

- Модерации аватарок пользователей
- Фильтрации постов в социальных сетях
- Защиты мессенджеров от неподобающего контента

## Формат данных

### Входные данные

- **Формат**: Изображения в формате JPEG
- **Обработка**: Изображения приводятся к размеру 200x200 пикселей и нормализуются
- **Предобработка**: Применяются трансформации (resize, normalization)

### Выходные данные

- **Выход**: Вероятность того, что изображение является NSFW (вещественное число из диапазона [0; 1])
- **Бинарная классификация**:
  - `1.0` - NSFW контент
  - `0.0` - SFW контент

## Метрики

Приоритетная задача - найти все NSFW картинки с минимальным количеством пропусков, поэтому основная метрика:

- **Recall (Полнота)** - способность модели найти все NSFW изображения
- **AUROC (Area Under ROC Curve)** - общая производительность классификатора
- **Accuracy** - точность классификации

Для базовой CNN модели используется Accuracy, для основной модели (ConvNext) - AUROC.

## Валидация и тест

- **Соотношение классов**: 1 к 1 (NSFW и SFW изображений)
- **Стратифицированное разбиение**:
  - Train: 80%
  - Val: 20%
  - Test: используется валидационный набор (можно расширить при необходимости)

Разбиение производится случайным образом с фиксированным seed для воспроизводимости результатов.

## Датасеты

Проект использует следующие датасеты:

1. **NSFW датасет**:
   - Источник: [HuggingFace - DarkyMan/nsfw-image-classification](https://huggingface.co/datasets/DarkyMan/nsfw-image-classification)
   - Содержит NSFW материалы в формате .jpg

2. **SFW датасет**:
   - Источник: [Mendeley - Human faces and object dataset](https://data.mendeley.com/datasets/nzwvnrmwp3/1)
   - Содержит изображения объектов и лиц

Данные хранятся с помощью DVC (Data Version Control) и находятся в `src/dataset/`.

## Моделирование

### Базовая модель (Baseline)

Обычная CNN архитектура с 4-5 скрытыми слоями:

- 2 сверточных слоя (Conv2d) с ReLU активацией
- MaxPooling слои для downsampling
- Полносвязные слои для классификации
- Binary Cross Entropy Loss

**Конфигурация**: `configs/config_baseline.yaml`

### Основная модель

**ConvNext-Tiny** - предобученная модель из библиотеки `timm`:

- Предобученный backbone (ImageNet)
- Замороженные нижние слои (transfer learning)
- Fine-tuning верхних слоев и классификатора
- LayerNorm и Dropout для регуляризации
- Binary Cross Entropy Loss with pos_weight для балансировки классов

**Конфигурация**: `configs/config.yaml`

### Обучение

- **Оптимизатор**: AdamW
- **Аугментации**: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur
- **Регуляризация**: Weight decay, Dropout

### Внедрение

Модель упакована в виде Python-пакета, который принимает путь к файлу изображения и возвращает предсказанный класс и вероятность.

## Структура проекта

```
nsfw/
├── configs/                 # Hydra конфигурации
│   ├── config.yaml          # Главный конфиг для ConvNext модели
│   ├── config_baseline.yaml # Конфиг для baseline CNN модели
│   ├── data.yaml            # Конфигурация данных
│   ├── model.yaml           # Параметры модели
│   ├── training.yaml        # Параметры обучения
│   ├── optimizer.yaml       # Параметры оптимизатора
│   ├── trainer.yaml         # Параметры PyTorch Lightning Trainer
│   ├── trainer_baseline.yaml # Trainer для baseline модели
│   ├── logging.yaml         # Настройки логирования (MLflow)
│   └── infer_config.yaml    # Конфигурация для инференса
├── src/
│   ├── dataset/             # Данные (управляются через DVC)
│   │   ├── nsfw/            # NSFW изображения
│   │   └── sfw/             # SFW изображения
│   └── nsfw/                # Python пакет
│       ├── __init__.py
│       ├── data.py          # Датасет классы
│       ├── model.py         # ConvNext модель
│       ├── baseline_model.py # CNN baseline модель
│       ├── train.py         # Скрипт обучения
│       ├── infer.py         # Скрипт инференса
│       └── conf_analyzer.py # Создание трансформаций из конфигов
├── checkpoints/             # Сохраненные модели
├── plots/                   # Графики и визуализации
├── .pre-commit-config.yaml  # Конфигурация pre-commit хуков
├── pyproject.toml           # Зависимости (Poetry)
├── poetry.lock              # Lock файл зависимостей
└── README.md                # Этот файл
```

## Setup

### Требования

- Python >= 3.13, < 3.15
- Poetry (для управления зависимостями)
- Git
- DVC (для управления данными)

### Установка зависимостей

1. **Клонируйте репозиторий**:
   ```bash
   git clone <repository-url>
   cd nsfw
   ```

2. **Установите Poetry** (если еще не установлен):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Установите зависимости проекта**:
   ```bash
   poetry install
   ```

4. **Установите dev зависимости** (для code quality tools):
   ```bash
   poetry install --with dev
   ```

5. **Активируйте виртуальное окружение Poetry**:
   ```bash
   poetry shell
   ```

   Или используйте команды через `poetry run`:
   ```bash
   poetry run python -m nsfw.train
   ```

### Настройка pre-commit

1. **Установите pre-commit хуки**:
   ```bash
   pre-commit install
   ```

2. **Проверьте все файлы**:
   ```bash
   pre-commit run -a
   ```

   Команда должна выполниться без ошибок (зеленый результат).

### Настройка данных

Данные управляются через DVC. Убедитесь, что DVC инициализирован:

```bash
# Инициализация DVC (если еще не сделано)
dvc init

# Скачивание данных (если данные в remote storage)
dvc pull
```

Если используется локальное хранилище, данные должны находиться в:
- `src/dataset/nsfw/` - NSFW изображения
- `src/dataset/sfw/` - SFW изображения

### Настройка MLflow

MLflow настроен на локальное хранилище (`.mlruns/`) по умолчанию. Для использования MLflow сервера измените `configs/logging.yaml`:

```yaml
mlflow:
  tracking_uri: "http://127.0.0.1:8080"  # Адрес MLflow сервера
  experiment_name: "nsfw_detection"
```

Для запуска локального MLflow сервера:
```bash
mlflow ui --port 8080
```

## Train

### Обучение основной модели (ConvNext)

Для обучения основной модели с предобученным ConvNext-Tiny:

```bash
python -m nsfw.train
```

Команда использует конфигурацию по умолчанию из `configs/config.yaml`.

### Обучение baseline модели (CNN)

Для обучения базовой CNN модели:

```bash
python -m nsfw.train --config-name=config_baseline
```

### Переопределение параметров через CLI

Вы можете переопределить любые параметры конфигурации через командную строку:

```bash
# Изменить batch size и количество эпох
python -m nsfw.train training.batch_size=64 training.max_epochs=20

# Изменить learning rate
python -m nsfw.train optimizer.learning_rate=0.001

# Использовать GPU
python -m nsfw.train trainer.accelerator=gpu trainer.devices=1

# Использовать конкретную GPU
python -m nsfw.train trainer.accelerator=gpu trainer.devices=[0]

# Комбинация параметров
python -m nsfw.train \
    training.batch_size=32 \
    training.max_epochs=15 \
    optimizer.learning_rate=0.0001 \
    trainer.accelerator=gpu
```

### Структура процесса обучения

1. **Загрузка данных**: Из `src/dataset/nsfw/` и `src/dataset/sfw/`
2. **Разделение данных**: Train/Val split согласно конфигурации
3. **Применение трансформаций**: Augmentation для train, базовые трансформации для val
4. **Создание DataLoader'ов**: Батчинг и загрузка данных
5. **Инициализация модели**: ConvNext или CNN baseline
6. **Обучение**: PyTorch Lightning Trainer
7. **Логирование**: Метрики и гиперпараметры в MLflow
8. **Сохранение чекпоинтов**: Лучшая модель сохраняется в `checkpoints/`

### Результаты обучения

- **Чекпоинты**: Сохраняются в `checkpoints/` (лучшая модель по val_auroc/val_accuracy)
- **Логи MLflow**: Все метрики, гиперпараметры и git commit ID логируются в MLflow
- **Модель в MLflow**: Автоматически сохраняется в MLflow artifact store

### Доступные метрики

Для ConvNext модели:
- `train_loss`, `val_loss` - функции потерь
- `train_auroc`, `val_auroc` - AUROC метрики

Для Baseline CNN модели:
- `train_loss`, `val_loss` - функции потерь
- `train_accuracy`, `val_accuracy` - Accuracy метрики

### Генерация графиков метрик

После обучения можно сгенерировать графики метрик из MLflow:

```bash
# Генерация графиков из последнего эксперимента (использует configs/plots_config.yaml)
python -m nsfw.generate_plots

# С указанием конкретного эксперимента
python -m nsfw.generate_plots mlflow.experiment_name=nsfw_detection

# С указанием конкретного run
python -m nsfw.generate_plots mlflow.run_id=<run_id>

# С указанием MLflow сервера
python -m nsfw.generate_plots mlflow.tracking_uri=http://127.0.0.1:8080

# Изменить директорию для сохранения графиков
python -m nsfw.generate_plots plots.output_dir=my_plots

# Комбинация параметров
python -m nsfw.generate_plots \
    mlflow.experiment_name=nsfw_detection \
    mlflow.tracking_uri=http://127.0.0.1:8080 \
    plots.output_dir=plots
```

Графики будут сохранены в директорию `plots/` (или указанную в конфиге):
- `loss_curves.png` - график train и val loss
- `auroc_curves.png` - график train и val AUROC
- `combined_metrics.png` - комбинированный график всех метрик

Конфигурация находится в `configs/plots_config.yaml` и может быть изменена там или через CLI параметры.

## Production Preparation

### Экспорт в ONNX

Для конвертации модели в ONNX формат (для продакшена):

```bash
# TODO: Добавить скрипт для экспорта в ONNX
# python -m nsfw.export_onnx --checkpoint checkpoints/best.ckpt --output model.onnx
```

ONNX модель может быть использована для инференса в различных средах (TensorFlow.js, ONNX Runtime, и т.д.).

### Экспорт в TensorRT

Для оптимизации модели для NVIDIA GPU с TensorRT:

```bash
# TODO: Добавить скрипт для экспорта в TensorRT
# python -m nsfw.export_tensorrt --onnx model.onnx --output model.trt
```

TensorRT обеспечивает значительное ускорение инференса на NVIDIA GPU.

### Артефакты для продакшена

Для развертывания модели в продакшене необходимы:

1. **Модель**: `.ckpt` файл (PyTorch Lightning) или `.onnx` / `.trt` (оптимизированные версии)
2. **Конфигурация**: Файлы из `configs/` (особенно `data.yaml` для трансформаций)
3. **Код инференса**: `src/nsfw/infer.py` и зависимые модули (`data.py`, `conf_analyzer.py`)
4. **Зависимости**: `pyproject.toml` или `requirements.txt`

Минимальный набор для инференса:
- `infer.py` - основной скрипт инференса
- `model.py` или `baseline_model.py` - определение модели
- `data.py` - загрузка данных
- `conf_analyzer.py` - создание трансформаций
- Конфигурационные файлы
- Чекпоинт модели

## Infer

### Формат входных данных

Для инференса модель ожидает:
- **Вход**: Директория с изображениями
- **Формат изображений**: JPEG (`.jpg` по умолчанию, можно изменить через `--pattern`)
- **Структура**: Все изображения в одной директории

### Запуск инференса

Базовый запуск с параметрами по умолчанию:

```bash
python -m nsfw.infer \
    --checkpoint checkpoints/best-epoch=05-val_auroc=0.92.ckpt \
    --input-dir data/test_images \
    --output predictions.csv
```

### Параметры инференса

Все параметры можно переопределить через CLI:

```bash
python -m nsfw.infer \
    checkpoint=checkpoints/best.ckpt \
    input_dir=data/test_images \
    output=predictions.csv \
    batch_size=64 \
    threshold=0.5 \
    pattern="*.jpg"
```

Или изменить в `configs/infer_config.yaml`:

```yaml
checkpoint: "checkpoints/best.ckpt"
input_dir: "data/test_images"
output: "predictions.csv"
batch_size: 32
threshold: 0.5
```

### Формат выходных данных

Результаты сохраняются в CSV или JSON формате:

**CSV формат**:
```csv
image_path,probability,prediction,class_name
data/test_images/img1.jpg,0.95,1,NSFW
data/test_images/img2.jpg,0.12,0,SFW
```

**JSON формат** (при указании `.json` расширения):
```json
[
  {
    "image_path": "data/test_images/img1.jpg",
    "probability": 0.95,
    "prediction": 1,
    "class_name": "NSFW"
  },
  {
    "image_path": "data/test_images/img2.jpg",
    "probability": 0.12,
    "prediction": 0,
    "class_name": "SFW"
  }
]
```

### Пример использования

1. **Подготовьте тестовые изображения**:
   ```bash
   mkdir -p data/test_images
   # Поместите изображения в data/test_images/
   ```

2. **Запустите инференс**:
   ```bash
   python -m nsfw.infer \
       checkpoint=checkpoints/best-epoch=05-val_auroc=0.92.ckpt \
       input_dir=data/test_images \
       output=results.csv
   ```

3. **Проверьте результаты**:
   ```bash
   head results.csv
   ```

### Пример данных

Примеры данных можно найти в датасетах:
- NSFW примеры: `src/dataset/nsfw/`
- SFW примеры: `src/dataset/sfw/`

Для тестирования можно использовать небольшую выборку из датасетов.

## Дополнительная информация

### Конфигурация проекта

Проект использует Hydra для управления конфигурациями. Все гиперпараметры вынесены в YAML файлы в `configs/`:

- `config.yaml` - главный конфиг для ConvNext модели
- `config_baseline.yaml` - конфиг для baseline CNN модели
- `data.yaml` - параметры данных и трансформаций
- `model.yaml` - архитектура модели
- `training.yaml` - параметры обучения
- `optimizer.yaml` - параметры оптимизатора
- `trainer.yaml` - параметры PyTorch Lightning
- `logging.yaml` - настройки MLflow

### Управление устройством (GPU/CPU)

По умолчанию используется `accelerator="auto"`, который автоматически определяет доступные устройства:

- **GPU** (CUDA) - если доступно
- **CPU** - если GPU нет
- **MPS** (Apple Silicon) - если доступно

Для явного указания устройства:

```bash
# Использовать GPU
python -m nsfw.train trainer.accelerator=gpu trainer.devices=1

# Использовать CPU
python -m nsfw.train trainer.accelerator=cpu

# Использовать конкретную GPU
python -m nsfw.train trainer.accelerator=gpu trainer.devices=[0]
```

Или через переменные окружения:

```bash
CUDA_VISIBLE_DEVICES=0 python -m nsfw.train
```

### Логирование экспериментов

Все эксперименты логируются в MLflow:
- Гиперпараметры (из Hydra конфигов)
- Метрики (loss, auroc/accuracy)
- Git commit ID
- Модель (артефакт)

Для просмотра результатов:

```bash
# Если используется локальное хранилище
mlflow ui

# Если используется сервер
mlflow ui --backend-store-uri http://127.0.0.1:8080
```

Откройте браузер по адресу `http://localhost:5000` для просмотра экспериментов.

## Автор

Букин Артемий Дмитриевич
