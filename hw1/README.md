# Домашнее задание 1: Тренировочный цикл и Linear Probe на ViT-Tiny

## Описание

Проект реализует полный цикл обучения нейронных сетей для классификации изображений на подмножестве CIFAR-10 (5 классов). Сравниваются две архитектуры:
- **SimpleCNN** - простая сверточная сеть (~2.5M параметров)
- **ViT-Tiny Linear Probe** - предобученный Vision Transformer с замороженным backbone (~5.5M параметров, 965 обучаемых)

## Структура проекта

```
hw1/
├── src/
│   ├── data.py          # Загрузка и подготовка данных
│   ├── models.py        # CNN и ViT-Tiny с linear probe
│   ├── train.py         # Тренировочный цикл
│   ├── evaluate.py      # Оценка и метрики
│   ├── profiling.py     # Профилировка моделей
│   └── utils.py         # Вспомогательные функции
├── scripts/
│   ├── train_cnn.py     # Скрипт обучения CNN
│   ├── train_vit.py     # Скрипт обучения ViT
│   └── profile_models.py # Скрипт профилировки
├── data/                # Данные CIFAR-10 (автоматически загружаются)
├── runs/                # TensorBoard логи
├── checkpoints/         # Сохраненные модели
├── results/             # Метрики, confusion matrices, профилировка
└── README.md
```

## Установка и запуск

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Обучение моделей

**CNN:**
```bash
python scripts/train_cnn.py --epochs 100 --batch_size 128 --lr 0.001
```

**ViT-Tiny:**
```bash
python scripts/train_vit.py --epochs 50 --batch_size 64 --lr 0.01
```

### Профилировка

```bash
python scripts/profile_models.py
```

### Просмотр логов TensorBoard

```bash
tensorboard --logdir runs/
```

## Описание архитектур

### SimpleCNN

Простая сверточная архитектура:
- 3 сверточных блока (Conv2d → BatchNorm → ReLU → MaxPool)
- 2 полносвязных слоя с Dropout (0.5)
- Инициализация весов: kaiming_normal
- Всего параметров: 2,471,941

**Архитектура:**
```
Input (3×32×32) 
→ Conv(64) → BN → ReLU → MaxPool → (64×16×16)
→ Conv(128) → BN → ReLU → MaxPool → (128×8×8)
→ Conv(256) → BN → ReLU → MaxPool → (256×4×4)
→ Flatten → FC(512) → Dropout → FC(5)
```

### ViT-Tiny Linear Probe

Предобученный Vision Transformer:
- Backbone: ViT-Tiny из timm (pretrained на ImageNet)
- Заморожены все слои кроме classification head
- Линейный классификатор: 192 → 5 классов
- Всего параметров: 5,525,381 (обучаемых: 965)

## Данные

- **Датасет:** CIFAR-10 (5 классов)
- **Классы:** airplane, automobile, bird, cat, dog
- **Размер:** 25,000 изображений (20,000 train + 5,000 test)
- **Train/Val split:** 80/20
- **Аугментации (train):**
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip(0.5)
  - Нормализация

## Эксперименты

### Гиперпараметры

**CNN:**
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 1e-4
- Batch size: 128
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)
- Early stopping: 15 epochs

**ViT-Tiny:**
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.01
- Weight decay: 1e-4
- Batch size: 64
- Scheduler: CosineAnnealingLR
- Early stopping: 10 epochs

### Sanity Checks

Обе модели прошли sanity check - способны переобучиться на маленьком батче из 32 сэмплов, достигнув 100% accuracy за 10-30 эпох.

## Результаты

### Метрики качества

| Модель | Test Accuracy | Macro F1 | Параметров (обучаемых) |
|--------|---------------|----------|------------------------|
| **CNN** | **91.02%** | **90.99%** | 2,471,941 |
| **ViT-Tiny** | 85.68% | 85.69% | 5,525,381 (965) |

**Победитель: CNN (+5.34% accuracy)**

### Per-class метрики (CNN)

| Класс | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| airplane | 0.915 | 0.936 | 0.925 |
| automobile | **0.972** | **0.988** | **0.980** |
| bird | 0.873 | 0.855 | 0.864 |
| cat | 0.881 | 0.875 | 0.878 |
| dog | 0.907 | 0.897 | 0.902 |

**Наблюдение:** Модель лучше всего распознает automobiles (98% F1), хуже всего - birds (86.4% F1).

### Per-class метрики (ViT-Tiny)

| Класс | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| airplane | 0.877 | 0.865 | 0.871 |
| automobile | **0.951** | **0.916** | **0.933** |
| bird | 0.837 | 0.797 | 0.817 |
| cat | 0.816 | 0.809 | 0.813 |
| dog | 0.810 | 0.897 | 0.851 |

### Confusion Matrix

Обе модели показывают похожие паттерны ошибок:
- Наибольшая путаница между cat/dog (внешне похожи)
- Bird часто путается с airplane (фон неба)
- Automobile распознается лучше всего (четкие контуры)

Confusion matrices сохранены в `results/confusion_matrix_cnn.png` и `results/confusion_matrix_vit.png`.

### Визуализации

**TensorBoard логи** содержат:
- Кривые обучения (train/val loss и accuracy)
- Learning rate scheduling
- Гистограммы весов и градиентов (каждые 5 эпох)

Для просмотра: `tensorboard --logdir runs/`

**Classification reports** с детальными метриками сохранены в `results/classification_report_*.txt`.

## Профилировка

### Производительность

| Метрика | CNN | ViT-Tiny | Преимущество |
|---------|-----|----------|--------------|
| Throughput | **11,059 img/s** | 1,742 img/s | CNN **6.3x быстрее** |
| Latency (batch=128/64) | **11.57 ms** | 36.73 ms | CNN **3.2x быстрее** |
| Peak Memory | **113 MB** | 209 MB | CNN **1.8x меньше** |
| Training time (to convergence) | 189s (98 epochs) | 160s (17 epochs) | Сопоставимо |

### Ключевые наблюдения

1. **CNN значительно быстрее** - throughput выше в 6.3 раза
2. **ViT требует больше памяти** - почти в 2 раза больше GPU memory
3. **ViT быстрее сходится** - early stopping на 17 эпохе vs 98 у CNN
4. **CNN более эффективен на малых изображениях** - 32×32 vs 224×224 для ViT

### Узкие места производительности

**CNN:**
- Основное время: convolutions (29%) и batch normalization (14%)
- Эффективные CUDA kernels от cuDNN
- Небольшой размер изображений (32×32) позволяет использовать большие батчи

**ViT:**
- Основное время: attention механизм (13.95% FMHA) и linear layers (55%)
- Большой размер входа (224×224) требует больше вычислений
- Self-attention квадратичен по количеству патчей

Профили сохранены в `runs/profiler/cnn/` и `runs/profiler/vit/`.

## Выводы

### 1. CNN vs ViT на малых данных

На малом датасете (25K изображений, 5 классов) **простая CNN превосходит ViT-Tiny linear probe** по всем метрикам:
- Выше точность: 91.02% vs 85.68% (+5.34%)
- Быстрее inference: 6.3x по throughput
- Меньше памяти: 113MB vs 209MB
- Лучше generalization на test set

### 2. Почему CNN оказался лучше?

**a) Размер данных**
- ViT требуют большие датасеты для обучения с нуля
- Linear probe ограничен возможностью адаптации: всего 965 параметров
- Предобучение на ImageNet не полностью переносится на low-resolution CIFAR-10

**b) Inductive bias**
- CNN встроенная locality и translation equivariance идеальны для изображений
- ViT требуют больше данных чтобы выучить эти свойства
- На малых разрешениях (32×32) CNN более эффективны

**c) Разрешение изображений**
- ViT upscale 32×32 → 224×224 теряет детали
- CNN работает напрямую с 32×32
- Upscaling добавляет артефакты интерполяции

### 3. Когда использовать CNN vs ViT?

**Используйте CNN когда:**
- Малый или средний датасет (<100K изображений)
- Ограниченные вычислительные ресурсы
- Требуется низкая латентность
- Работа с low-resolution изображениями

**Используйте ViT когда:**
- Большой датасет (>1M изображений)
- Достаточно вычислительных ресурсов
- High-resolution изображения (224×224+)
- Нужна transfer learning способность
- Важна глобальная контекстная информация

### 4. Скорость сходимости

Интересно, что **ViT сошелся быстрее** (17 vs 98 эпох), несмотря на худшее качество:
- Предобученный backbone уже содержит хорошие признаки
- Linear probe требует минимальной адаптации
- Early stopping сработал раньше из-за меньшей емкости модели

### 5. Практические рекомендации

1. **Для production на CIFAR-подобных задачах** - используйте современные CNN (ResNet, EfficientNet)
2. **Для research с большими данными** - ViT показывают лучшие результаты на ImageNet-scale
3. **Для limited resources** - CNN более эффективны по параметрам/FLOPs
4. **Для transfer learning** - ViT linear probe требует меньше обучения, но может давать lower accuracy

## Дополнительные материалы

- **TensorBoard логи:** `runs/cnn/` и `runs/vit/`
- **Профили производительности:** `runs/profiler/cnn_trace.json` и `runs/profiler/vit_trace.json`
- **Confusion matrices:** `results/confusion_matrix_*.png`
- **Сохраненные модели:** `checkpoints/cnn_best.pth` и `checkpoints/vit_best.pth`
- **Детальные метрики:** `results/per_class_metrics_*.csv`

## Воспроизводимость

Все эксперименты воспроизводимы с фиксированными random seeds (42):
- `torch.manual_seed(42)`
- `numpy.random.seed(42)`
- `random.seed(42)`
- `torch.backends.cudnn.deterministic = True`

## Требования

- Python 3.8+
- PyTorch 2.0+
- CUDA (опционально, но рекомендуется)
- 8GB RAM минимум
- 2GB GPU memory минимум для CNN, 4GB для ViT

---

## Краткая сводка результатов

| Характеристика | CNN | ViT-Tiny | Лучший |
|----------------|-----|----------|--------|
| **Test Accuracy** | 91.02% | 85.68% | **CNN** |
| **Throughput** | 11,059 img/s | 1,742 img/s | **CNN (6.3x)** |
| **Latency** | 11.57 ms | 36.73 ms | **CNN (3.2x)** |
| **Memory** | 113 MB | 209 MB | **CNN (1.8x)** |
| **Convergence** | 98 epochs | 17 epochs | **ViT** |
| **Parameters** | 2.5M | 5.5M (965 trainable) | **CNN** |

**Вывод:** На малых датасетах с low-resolution изображениями CNN превосходит ViT-Tiny linear probe по всем ключевым метрикам.
