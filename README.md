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

**Конфигурация**: `configs/train/config_baseline.yaml`

### Основная модель

**ConvNext-Tiny** - предобученная модель из библиотеки `timm`:

- Предобученный backbone (ImageNet)
- Замороженные нижние слои (transfer learning)
- Fine-tuning верхних слоев и классификатора
- LayerNorm и Dropout для регуляризации
- Binary Cross Entropy Loss with pos_weight для балансировки классов

**Конфигурация**: `configs/train/config.yaml`

### Обучение

- **Оптимизатор**: AdamW
- **Аугментации**: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur
- **Регуляризация**: Weight decay, Dropout

### Внедрение

Модель упакована в виде Python-пакета и доступна через несколько интерфейсов:

1. **Локальный инференс** (`nsfw.infer`): Скрипт для пакетной обработки изображений из директории
2. **MLflow Serving** (`nsfw.serve_model`): REST API сервер для продакшена с автоматическим препроцессингом
3. **ONNX Runtime** (`nsfw.export_onnx`): Экспортированная ONNX модель для высокопроизводительного инференса

Все методы принимают путь к файлу изображения и возвращают предсказанный класс и вероятность.

## Структура проекта

```
nsfw/
├── configs/                 # Hydra конфигурации
│   ├── train/               # Конфигурации для обучения
│   │   ├── config.yaml      # Главный конфиг для ConvNext модели
│   │   └── config_baseline.yaml # Конфиг для baseline CNN модели
│   ├── data/                # Конфигурация данных
│   │   └── data.yaml
│   ├── model/               # Параметры моделей
│   │   ├── model.yaml
│   │   └── baseline_model.yaml
│   ├── training/            # Параметры обучения
│   │   └── training.yaml
│   ├── optimizer/           # Параметры оптимизатора
│   │   └── optimizer.yaml
│   ├── trainer/             # Параметры PyTorch Lightning Trainer
│   │   ├── trainer.yaml
│   │   └── trainer_baseline.yaml
│   ├── logging/             # Настройки логирования (MLflow)
│   │   └── logging.yaml
│   ├── infer/               # Конфигурация для локального инференса
│   │   └── infer_config.yaml
│   ├── export/              # Конфигурация для экспорта в ONNX
│   │   └── export_onnx_config.yaml
│   ├── serve/               # Конфигурация для MLflow Serving
│   │   └── serve_model_config.yaml
│   ├── register/            # Конфигурация для регистрации модели
│   │   └── register_model_config.yaml
│   ├── plots/               # Конфигурация для генерации графиков
│   │   └── plots_config.yaml
│   └── mlflow_model/        # Конфигурация для MLflow модели
│       └── mlflow_model_config.yaml # Параметры препроцессинга
├── src/
│   ├── dataset/             # Данные (управляются через DVC)
│   │   ├── nsfw/            # NSFW изображения
│   │   ├── sfw/             # SFW изображения
│   │   ├── nsfw.dvc         # DVC файл для NSFW данных
│   │   └── sfw.dvc          # DVC файл для SFW данных
│   └── nsfw/                # Python пакет
│       ├── __init__.py
│       ├── data.py          # Датасет классы и загрузка данных
│       ├── model.py         # ConvNext модель
│       ├── baseline_model.py # CNN baseline модель
│       ├── train.py         # Скрипт обучения
│       ├── infer.py         # Скрипт локального инференса
│       ├── conf_analyzer.py # Создание трансформаций из конфигов
│       ├── dvc_utils.py     # Утилиты для работы с DVC
│       ├── export_onnx.py   # Экспорт модели в ONNX
│       ├── generate_plots.py # Генерация графиков метрик
│       ├── mlflow_model.py  # MLflow модель с препроцессингом
│       ├── register_model.py # Регистрация модели в MLflow
│       └── serve_model.py   # Запуск MLflow Serving сервера
├── checkpoints/             # Сохраненные модели
├── plots/                   # Графики и визуализации метрик
├── models/                  # Экспортированные модели (ONNX)
├── .mlruns/                 # MLflow эксперименты (локальное хранилище)
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

Данные управляются через DVC и автоматически загружаются при запуске обучения.

**Автоматическая загрузка:**
При запуске `train.py` данные автоматически загружаются через DVC API. Если DVC недоступен, используется функция `download_data()` для загрузки из открытых источников.

**Ручная загрузка через DVC:**
```bash
# Инициализация DVC (если еще не сделано)
dvc init

# Скачивание данных (если данные в remote storage)
dvc pull
```

**Локальное хранилище:**
Если используется локальное хранилище, данные должны находиться в:
- `src/dataset/nsfw/` - NSFW изображения
- `src/dataset/sfw/` - SFW изображения

Если данные отсутствуют, они будут автоматически загружены из открытых источников при первом запуске обучения.

### Настройка MLflow

MLflow настроен на локальное хранилище (`.mlruns/`) по умолчанию. Для использования MLflow сервера измените `configs/logging/logging.yaml`:

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

Команда использует конфигурацию по умолчанию из `configs/train/config.yaml`.

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
# Генерация графиков из последнего эксперимента (использует configs/plots/plots_config.yaml)
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

Конфигурация находится в `configs/plots/plots_config.yaml` и может быть изменена там или через CLI параметры.

## Production Preparation

### Экспорт в ONNX

Для конвертации модели в ONNX формат (для продакшена):

```bash
# Базовое использование (из configs/export/export_onnx_config.yaml)
python -m nsfw.export_onnx

# С указанием конкретного чекпоинта
python -m nsfw.export_onnx \
    checkpoint=checkpoints/best-epoch=07-val_auroc=1.00.ckpt \
    output=models/model.onnx

# С настройкой параметров экспорта
python -m nsfw.export_onnx \
    checkpoint=checkpoints/best.ckpt \
    output=models/model.onnx \
    input_size=[200,200] \
    opset_version=14 \
    dynamic_batch=true
```

ONNX модель может быть использована для инференса в различных средах (TensorFlow.js, ONNX Runtime, и т.д.).

**Параметры экспорта:**
- `checkpoint` - путь к чекпоинту модели
- `output` - путь для сохранения ONNX модели
- `input_size` - размер входного изображения [height, width]
- `opset_version` - версия ONNX opset (рекомендуется 14)
- `dynamic_batch` - использовать ли динамический batch size

### Экспорт в TensorRT

TensorRT экспорт не реализован в данном проекте. При необходимости можно добавить экспорт из ONNX модели в TensorRT для оптимизации инференса на NVIDIA GPU.

**Примечание:** Для продакшена рекомендуется использовать MLflow Serving (см. раздел [Infer](#infer)), который обеспечивает удобное API и автоматический препроцессинг.

### Артефакты для продакшена

Для развертывания модели в продакшене доступны два варианта:

#### Вариант 1: MLflow Serving (Рекомендуется)

Модель регистрируется в MLflow Model Registry с автоматическим препроцессингом:
- Модель: зарегистрирована в MLflow Model Registry
- Препроцессинг: встроен в модель через `mlflow_model.py`
- API: REST API через MLflow Serving
- Конфигурация: загружается автоматически из `configs/train/config.yaml`

**Преимущества:**
- Автоматический препроцессинг
- Версионирование моделей
- Простое развертывание
- REST API из коробки

#### Вариант 2: Локальный инференс

Для локального использования без сервера:

1. **Модель**: `.ckpt` файл (PyTorch Lightning) или `.onnx` (оптимизированная версия)
2. **Конфигурация**: Файлы из `configs/` (особенно `data.yaml` для трансформаций)
3. **Код инференса**: `src/nsfw/infer.py` и зависимые модули
4. **Зависимости**: `pyproject.toml` или `requirements.txt`

Минимальный набор для локального инференса:
- `infer.py` - основной скрипт инференса
- `model.py` или `baseline_model.py` - определение модели
- `data.py` - загрузка данных
- `conf_analyzer.py` - создание трансформаций
- Конфигурационные файлы из `configs/` (структурированы по папкам)
- Чекпоинт модели

## Infer

### Формат входных данных

Для инференса модель ожидает:
- **Вход**: Директория с изображениями
- **Формат изображений**: JPEG (`.jpg` по умолчанию, можно изменить через `--pattern`)
- **Структура**: Все изображения в одной директории

### Запуск инференса (локально)

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

### MLflow Serving (Inference Server)

Для развертывания модели через MLflow Serving:

#### 1. Регистрация и запуск сервера

```bash
# Регистрация модели с препроцессингом и запуск сервера
python -m nsfw.serve_model \
    checkpoint=checkpoints/best-epoch=07-val_auroc=1.00.ckpt \
    model_name=nsfw_detection_model \
    serving.host=127.0.0.1 \
    serving.port=5000
```

Эта команда:
- Регистрирует модель в MLflow Model Registry (версионирование автоматически)
- Загружает конфигурацию из `configs/train/config.yaml` для инициализации модели
- Автоматически запускает serving сервер на указанном порту

**Примечание:** Конфигурация модели автоматически загружается из `configs/train/config.yaml` при отсутствии гиперпараметров в чекпоинте.

#### 2. Запуск сервера вручную

Если модель уже зарегистрирована:

```bash
# Запуск сервера для последней версии модели
mlflow models serve -m models:/nsfw_detection_model/latest \
    --host 127.0.0.1 \
    --port 5000 \
    --no-conda
```

#### 3. Использование API

После запуска сервера модель доступна через REST API. Модель принимает на вход **массив пикселей изображения** (numpy array).

**Формат входных данных:**
- Массив пикселей изображения в формате `[height, width, channels]`
- Значения пикселей в диапазоне `[0, 255]` (uint8)
- Поддерживаются RGB (3 канала) и Grayscale (1 канал, автоматически конвертируется в RGB)

**Предсказание для изображения:**

```python
import requests
import numpy as np
from PIL import Image
import json

# Загружаем изображение и конвертируем в numpy array
image = Image.open("path/to/image.jpg").convert("RGB")
image_array = np.array(image)  # Shape: [H, W, 3], значения [0, 255]

# Отправка запроса
url = "http://127.0.0.1:5000/invocations"
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps({"inputs": image_array.tolist()})
)

result = response.json()
print(result)
```

**Ответ:**
```json
{
  "probability": 0.95,
  "prediction": 1,
  "class": "NSFW"
}
```

**Важно:**
- Входные данные должны быть массивом пикселей, а не путем к файлу
- Формат: `[height, width, channels]`
- Значения пикселей должны быть в диапазоне `[0, 255]` (uint8)
- Сервер автоматически применяет препроцессинг (resize, normalization) согласно конфигурации из `configs/mlflow_model/mlflow_model_config.yaml`

#### 4. Использование Python клиента

```python
import requests
import numpy as np
from PIL import Image
import json

# URL сервера
url = "http://127.0.0.1:5000/invocations"

# Загружаем изображение и конвертируем в numpy array
image = Image.open("path/to/image.jpg").convert("RGB")
image_array = np.array(image)  # Shape: [H, W, 3], значения [0, 255]

# Отправка запроса
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    data=json.dumps({"inputs": image_array.tolist()})
)

# Получение результата
result = response.json()
print(result)
# Output: {"probability": 0.95, "prediction": 1, "class": "NSFW"}
```

#### 5. Конфигурация сервера

Настройки сервера можно изменить в `configs/serve/serve_model_config.yaml`:

```yaml
serving:
  host: "127.0.0.1"  # Адрес сервера
  port: 5000          # Порт сервера
```

Или через CLI:

```bash
python -m nsfw.serve_model \
    serving.host=0.0.0.0 \
    serving.port=8080
```

#### 6. Остановка сервера

Для остановки сервера:
```bash
# Если сервер запущен в терминале
Ctrl+C

# Или через kill процесса
pkill -f "mlflow models serve"
pkill -f "uvicorn.*mlflow.pyfunc.scoring_server"
```

#### 7. Проверка здоровья сервера

```bash
curl http://127.0.0.1:5000/health
```

Должен вернуть `200 OK` если сервер работает.

#### 8. Перезапуск сервера

Для перезапуска сервера после изменений в коде:
```bash
# Остановить текущий сервер
pkill -f "mlflow models serve"

# Перезапустить
python -m nsfw.serve_model
```

**Примечание:** После изменений в `mlflow_model.py` необходимо перерегистрировать модель в MLflow Model Registry (новая версия будет создана автоматически).

Или изменить в `configs/infer/infer_config.yaml`:

```yaml
checkpoint: "checkpoints/best.ckpt"
input_dir: "data/test_images"
output: "predictions.csv"
batch_size: 32
threshold: 0.5
```

### Сравнение методов инференса

| Метод | Преимущества | Недостатки | Использование |
|-------|-------------|------------|---------------|
| **MLflow Serving** | REST API, версионирование, автоматический препроцессинг | Требует запущенный сервер | Продакшен, микросервисы |
| **Локальный инференс** | Простота, не требует сервера | Нужно управлять зависимостями | Разовые задачи, разработка |
| **ONNX Runtime** | Высокая производительность, кроссплатформенность | Требует экспорт модели | Edge устройства, мобильные приложения |

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

Проект использует Hydra для управления конфигурациями. Все гиперпараметры вынесены в YAML файлы в `configs/`, структурированные по папкам:

**Основные конфигурации:**
- `train/config.yaml` - главный конфиг для ConvNext модели
- `train/config_baseline.yaml` - конфиг для baseline CNN модели
- `data.yaml` - параметры данных и трансформаций
- `model.yaml` - архитектура модели
- `training.yaml` - параметры обучения
- `optimizer.yaml` - параметры оптимизатора
- `trainer.yaml` - параметры PyTorch Lightning
- `logging.yaml` - настройки MLflow

**Конфигурации для инференса и продакшена:**
- `infer/infer_config.yaml` - конфигурация для локального инференса
- `export/export_onnx_config.yaml` - параметры экспорта в ONNX
- `serve/serve_model_config.yaml` - настройки MLflow Serving сервера
- `register/register_model_config.yaml` - параметры регистрации модели
- `plots/plots_config.yaml` - настройки генерации графиков
- `mlflow_model/mlflow_model_config.yaml` - параметры препроцессинга для MLflow модели

Все конфигурации можно переопределить через CLI параметры Hydra.

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

### Доступные скрипты

Проект включает следующие скрипты:

- **`nsfw.train`** - Обучение модели (ConvNext или baseline CNN)
- **`nsfw.infer`** - Локальный инференс на пакете изображений
- **`nsfw.export_onnx`** - Экспорт модели в ONNX формат
- **`nsfw.serve_model`** - Регистрация модели и запуск MLflow Serving сервера
- **`nsfw.register_model`** - Регистрация модели в MLflow Model Registry (без запуска сервера)
- **`nsfw.generate_plots`** - Генерация графиков метрик из MLflow экспериментов

Все скрипты используют Hydra для конфигурации и поддерживают переопределение параметров через CLI.

### Технические детали

**Управление данными:**
- Данные управляются через DVC (Data Version Control)
- Автоматическая загрузка данных при запуске обучения/инференса через `dvc_utils.py`
- Fallback на `download_data()` при отсутствии DVC

**Конфигурация:**
- Все гиперпараметры вынесены в YAML конфиги через Hydra
- Конфигурация модели автоматически загружается из `configs/train/config.yaml` при отсутствии в чекпоинте
- Поддержка иерархических конфигов

**Логирование:**
- MLflow для трекинга экспериментов
- Автоматическое логирование метрик, гиперпараметров и git commit ID
- Генерация графиков метрик из MLflow данных через `generate_plots.py`

**Code Quality:**
- Pre-commit хуки: black, isort, flake8, prettier
- Все хуки настроены и проходят проверку

## Автор

Букин Артемий Дмитриевич
