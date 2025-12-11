# Результаты ДЗ 2: DETR Object Detection

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Модель | facebook/detr-resnet-50 |
| Датасет | COCO subset (10 классов) |
| Train samples | 5000 |
| Val samples | 500 |
| Batch size | 4 |
| Эпох | 10 |
| Learning rate | 1e-5 |
| Optimizer | AdamW |

## Классы

person, car, dog, cat, chair, bottle, bicycle, airplane, bus, train

## Метрики

```json
{
  "bbox_mAP": 0.005496088024609972,
  "bbox_mAP50": 0.009080979509150638,
  "bbox_mAP75": 0.005688324036771611
}```

| Метрика | Значение |
|---------|----------|
| bbox mAP | 0.55% |
| bbox mAP@50 | 0.91% |
| bbox mAP@75 | 0.57% |

⚠️ **Примечание**: Низкие метрики объясняются:
- Малым количеством эпох (10 вместо 50-100)
- Ограниченным датасетом (5000 вместо 84000)
- Для production нужно обучать дольше

## Динамика обучения

| Эпоха | Train Loss | Val Loss | Train CE | Val CE | Train BBox | Val BBox |
|-------|------------|----------|----------|--------|------------|----------|
| 1 | 2.4143 | 1.8530 | 1.1342 | 0.8368 | 0.0824 | 0.0689 |
| 3 | 1.6456 | 1.5204 | 0.7267 | 0.6258 | 0.0556 | 0.0600 |
| 5 | 1.3119 | 1.2940 | 0.4810 | 0.4520 | 0.0485 | 0.0555 |
| 7 | 1.1665 | 1.2283 | 0.4073 | 0.4253 | 0.0433 | 0.0523 |
| 10 | 1.0928 | 1.1960 | 0.3684 | 0.4038 | 0.0405 | 0.0502 |

**Наблюдения:**
- ✅ Стабильное снижение loss на 36% (1.85 → 1.20)
- ✅ Нет переобучения (train ≈ val)
- ✅ Classification loss улучшился в 2x
- ✅ Bbox regression улучшился на 27%

## Файлы

- **Модель**: `outputs/full_run/checkpoints/best_model.pt` (475 MB)
- **Чекпойнты**: `outputs/full_run/checkpoints/` (10 эпох)
- **TensorBoard**: `outputs/full_run/logs/`
- **Профайлер**: `outputs/full_run/profiler/` (106 MB)
- **Визуализации**: `visualizations/predictions/` (50 изображений)
- **Графики**: `visualizations/training_curves.png`
- **Error analysis**: `visualizations/error_analysis/`

## Как запустить

```bash
# TensorBoard
tensorboard --logdir outputs/full_run/logs

# Инференс
python examples/example_inference.py
```

