#!/bin/bash

# Full Pipeline - полный цикл обучения и анализа

echo "=========================================="
echo "DETR Object Detection - Full Pipeline"
echo "=========================================="
echo ""

# Параметры
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-10}
NUM_WORKERS=${NUM_WORKERS:-4}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-5000}
MAX_VAL_SAMPLES=${MAX_VAL_SAMPLES:-500}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/full_run"}

echo "Параметры:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Workers: $NUM_WORKERS"
echo "  Train Samples: $MAX_TRAIN_SAMPLES"
echo "  Val Samples: $MAX_VAL_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Активация venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Проверка COCO
if [ ! -d "data/coco/train2017" ]; then
    echo "❌ COCO dataset не найден! Запустите ./download_coco.sh"
    exit 1
fi

echo "=========================================="
echo "Этап 1/6: Обучение DETR"
echo "=========================================="
echo ""

python src/train.py \
    --data_dir ./data/coco \
    --output_dir "$OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --num_workers $NUM_WORKERS \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_val_samples $MAX_VAL_SAMPLES \
    --profile_epoch 2

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при обучении!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Этап 2/6: Оценка модели (mAP)"
echo "=========================================="
echo ""

python src/evaluate.py \
    --data_dir ./data/coco \
    --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt" \
    --config "$OUTPUT_DIR/config.json" \
    --output "$OUTPUT_DIR/metrics.json" \
    --num_workers $NUM_WORKERS

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при оценке!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Этап 3/6: Анализ ошибок"
echo "=========================================="
echo ""

python src/error_analysis.py \
    --data_dir ./data/coco \
    --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt" \
    --config "$OUTPUT_DIR/config.json" \
    --output_dir ./visualizations/error_analysis \
    --num_workers $NUM_WORKERS

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при анализе ошибок!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Этап 4/6: Визуализация предсказаний"
echo "=========================================="
echo ""

python src/visualize.py \
    --data_dir ./data/coco \
    --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt" \
    --config "$OUTPUT_DIR/config.json" \
    --output_dir ./visualizations/predictions \
    --num_images 50

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при визуализации!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Этап 5/6: Построение графиков обучения"
echo "=========================================="
echo ""

# Найти последнюю директорию с логами
LATEST_LOG=$(ls -t "$OUTPUT_DIR/logs" | head -1)

if [ ! -z "$LATEST_LOG" ]; then
    python src/visualize.py \
        --checkpoint "$OUTPUT_DIR/checkpoints/best_model.pt" \
        --config "$OUTPUT_DIR/config.json" \
        --plot_curves \
        --log_dir "$OUTPUT_DIR/logs/$LATEST_LOG" \
        --output_dir ./visualizations
else
    echo "⚠ Логи TensorBoard не найдены"
fi

echo ""
echo "=========================================="
echo "Этап 6/6: Генерация отчета"
echo "=========================================="
echo ""

# Создание сводного отчета
cat > "$OUTPUT_DIR/summary.txt" << EOF
========================================
DETR Object Detection - Сводка обучения
========================================

Дата: $(date)

Параметры обучения:
- Batch Size: $BATCH_SIZE
- Epochs: $NUM_EPOCHS
- Model: facebook/detr-resnet-50

Метрики (см. metrics.json):
$(cat "$OUTPUT_DIR/metrics.json" 2>/dev/null || echo "Не найдено")

Файлы:
- Модель: $OUTPUT_DIR/checkpoints/best_model.pt
- Конфиг: $OUTPUT_DIR/config.json
- Метрики: $OUTPUT_DIR/metrics.json
- TensorBoard: $OUTPUT_DIR/logs/
- Профайлер: $OUTPUT_DIR/profiler/

Визуализации:
- Ошибки: ./visualizations/error_analysis/
- Предсказания: ./visualizations/predictions/
- Графики: ./visualizations/training_curves.png

Для просмотра в TensorBoard:
tensorboard --logdir $OUTPUT_DIR/logs

========================================
EOF

cat "$OUTPUT_DIR/summary.txt"

echo ""
echo "✅ Pipeline завершен успешно!"
echo ""
echo "Результаты сохранены в: $OUTPUT_DIR"
echo "Визуализации в: ./visualizations"
echo ""

