# DETR Object Detection + Synthetic Data

Полная реализация ДЗ 2 & 2.5: обучение DETR на COCO subset + генерация синтетических данных.

## 🚀 Быстрый старт

```bash
# 1. Установка
./setup.sh

# 2. Загрузка COCO dataset (~20GB)
./download_coco.sh

# 3. Быстрый тест (5-10 минут)
./quick_start.sh

# 4. TensorBoard
source venv/bin/activate
tensorboard --logdir outputs
```

## 📁 Структура

```
├── src/                        # Исходный код
│   ├── dataset.py             # COCO dataset loader
│   ├── train.py               # Обучение DETR
│   ├── evaluate.py            # Оценка (mAP)
│   ├── error_analysis.py      # Анализ ошибок
│   ├── visualize.py           # Визуализация
│   ├── generate_synthetic.py  # Stable Diffusion + ControlNet
│   └── ablation_study.py      # Сравнение с/без синтетики
├── data/coco/                 # COCO dataset
├── outputs/                   # Результаты экспериментов
├── visualizations/            # Визуализации предсказаний
└── examples/                  # Примеры использования
```

## 📚 Основные команды

### Обучение

```bash
source venv/bin/activate

# Быстрый тест (2 эпохи, 50 train samples)
python src/train.py \
    --output_dir ./outputs/test \
    --num_epochs 2 \
    --max_train_samples 50 \
    --max_val_samples 20

# Полное обучение (10 эпох, 5000 train samples)
python src/train.py \
    --output_dir ./outputs/full \
    --num_epochs 10 \
    --batch_size 4 \
    --max_train_samples 5000 \
    --max_val_samples 500

# Или через скрипт (автоматически ограничивает датасет)
./full_pipeline.sh

# Для полного COCO (долго, ~20 часов)
python src/train.py \
    --output_dir ./outputs/full_coco \
    --num_epochs 10 \
    --batch_size 4
```

**Параметры:**
- `--batch_size 4` - размер батча (уменьшите до 2 при OOM)
- `--num_epochs 10` - количество эпох
- `--lr 1e-5` - learning rate
- `--profile_epoch 2` - эпоха для профайлера

### Визуализация предсказаний

```bash
python src/visualize.py \
    --checkpoint ./outputs/quick_test/checkpoints/best_model.pt \
    --config ./outputs/quick_test/config.json \
    --num_images 20
```

### Генерация синтетических данных

```bash
python src/generate_synthetic.py \
    --output_dir ./data/synthetic \
    --classes dog cat \
    --num_samples 50
```

### Ablation Study

```bash
python src/ablation_study.py \
    --output_dir ./outputs/ablation \
    --quick_test  # для быстрой проверки
```

## 📊 Результаты

После обучения вы получите:

**Модель и метрики:**
- `outputs/{exp}/checkpoints/best_model.pt` - обученная модель
- `outputs/{exp}/config.json` - конфигурация
- `outputs/{exp}/logs/` - TensorBoard логи
- `outputs/{exp}/profiler/` - trace профайлера

**Визуализации:**
- `visualizations/predictions/` - Ground Truth vs Predictions
- `visualizations/error_analysis/` - анализ ошибок (если запущен)

**Пример результатов (2 эпохи quick test):**
```
Epoch 1: Train Loss 4.09 → Val Loss 3.59
Epoch 2: Train Loss 3.80 → Val Loss 3.40
```

## 🎯 Выполнение ДЗ

### Задание 2: DETR Object Detection

**Что реализовано:**
- ✅ COCO subset (10 классов): person, car, dog, cat, chair, bottle, bicycle, airplane, bus, train
- ✅ Fine-tuning DETR ResNet-50
- ✅ TensorBoard логирование (loss, loss_ce, loss_bbox, loss_giou)
- ✅ Профайлер (запускается на эпохе 2)
- ✅ Сохранение чекпойнтов
- ✅ Визуализация предсказаний

**Для сдачи нужно:**
1. Запустить полное обучение (10 эпох)
2. Доработать evaluation для подсчета mAP на subset
3. Запустить error analysis

### Задание 2.5: Synthetic Data

**Что реализовано:**
- ✅ Код генерации через Stable Diffusion + ControlNet
- ✅ Ablation study скрипт
- ✅ Код обучения с интеграцией синтетики

**Для сдачи нужно:**
1. Сгенерировать синтетические данные
2. Обучить 2 модели (с/без синтетики)
3. Сравнить метрики

## 💾 Гиперпараметры

| Параметр | Значение |
|----------|----------|
| Model | facebook/detr-resnet-50 |
| Classes | 10 (COCO subset) |
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| LR Schedule | StepLR (γ=0.1, step=5) |
| Gradient Clip | 0.1 |
| Epochs | 10 |

## 🛠 Примеры

### Инференс на одном изображении

```python
from examples.example_inference import load_model, detect_objects, visualize_detections

# Загрузить модель
model, processor, classes = load_model(
    checkpoint='./outputs/quick_test/checkpoints/best_model.pt',
    config='./outputs/quick_test/config.json'
)

# Детекция
image, results = detect_objects('path/to/image.jpg', model, processor, classes)

# Визуализация
visualize_detections(image, results, save_path='result.jpg')
```

### Batch инференс

```bash
python examples/example_batch_inference.py
```

## 🔧 Устранение проблем

**Out of Memory:**
```bash
python src/train.py --batch_size 2
```

**COCO dataset не найден:**
```bash
./download_coco.sh
# Или укажите путь: --data_dir /path/to/coco
```

**Медленное обучение:**
- Используйте GPU
- Увеличьте `--num_workers`
- Проверьте `nvidia-smi`

## 📋 Требования

**Минимальные:**
- GPU: 6GB VRAM
- RAM: 16GB
- Диск: 30GB

**Для синтетики:**
- GPU: 8GB+ VRAM (Stable Diffusion)

## 📖 Документация кода

### src/train.py

Обучение с автоматическим логированием и профилированием.

```bash
python src/train.py --help
```

### src/visualize.py

Визуализация + построение графиков loss.

```bash
python src/visualize.py \
    --checkpoint path/to/model.pt \
    --config path/to/config.json \
    --plot_curves \
    --log_dir path/to/logs
```

### src/generate_synthetic.py

Генерация синтетических данных для редких классов.

```bash
python src/generate_synthetic.py \
    --classes dog cat \
    --num_samples 100
```

## 🎓 Чек-лист сдачи

**ДЗ 2:**
- [ ] Обучена DETR на 10 классах
- [ ] TensorBoard логи
- [ ] Trace профайлера
- [ ] Таблица метрик (mAP)
- [ ] Визуализации боксов
- [ ] Error analysis

**ДЗ 2.5:**
- [ ] Сгенерированы синтетические данные
- [ ] Обучены 2 модели (baseline vs +synthetic)
- [ ] Таблица сравнения
- [ ] Визуализации синтетики

## 🔗 Ссылки

- [DETR Paper](https://arxiv.org/abs/2005.12872)
- [Hugging Face DETR](https://huggingface.co/facebook/detr-resnet-50)
- [COCO Dataset](https://cocodataset.org/)

---

**Автор:** ДЗ 2 & 2.5 - Computer Vision Course  
**Дата:** 2025-11-29
