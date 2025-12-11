#!/bin/bash

# Quick Start Script - запускает весь pipeline с тестовыми параметрами

echo "=========================================="
echo "DETR Object Detection - Quick Start"
echo "=========================================="
echo ""

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "❌ Виртуальное окружение не найдено!"
    echo "Создание виртуального окружения..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    echo "✓ Активация виртуального окружения..."
    source venv/bin/activate
fi

echo ""

# Проверка COCO dataset
if [ ! -d "data/coco/train2017" ]; then
    echo "❌ COCO dataset не найден!"
    echo "Запустите: ./download_coco.sh для загрузки датасета"
    echo "Или используйте --max_train_samples для тестирования на малой выборке"
    exit 1
else
    echo "✓ COCO dataset найден"
fi

echo ""
echo "=========================================="
echo "Шаг 1: Обучение DETR модели (тестовый режим)"
echo "=========================================="
echo ""

python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/quick_test \
    --batch_size 2 \
    --num_epochs 2 \
    --max_train_samples 50 \
    --max_val_samples 20 \
    --profile_epoch 1

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при обучении!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Шаг 2: Оценка модели (пропущена для quick test)"
echo "=========================================="
echo ""
echo "⚠ Evaluation пропущена в quick test."
echo "  Для полной оценки используйте:"
echo "  python src/evaluate.py --checkpoint ./outputs/quick_test/checkpoints/best_model.pt --config ./outputs/quick_test/config.json"
echo ""

echo ""
echo "=========================================="
echo "Шаг 3: Анализ ошибок (пропущен для quick test)"
echo "=========================================="
echo ""
echo "⚠ Error analysis пропущен в quick test."
echo "  Для анализа используйте:"
echo "  python src/error_analysis.py --checkpoint ./outputs/quick_test/checkpoints/best_model.pt --config ./outputs/quick_test/config.json"
echo ""

echo ""
echo "=========================================="
echo "Шаг 4: Визуализация предсказаний"
echo "=========================================="
echo ""

python src/visualize.py \
    --data_dir ./data/coco \
    --checkpoint ./outputs/quick_test/checkpoints/best_model.pt \
    --config ./outputs/quick_test/config.json \
    --output_dir ./visualizations/quick_test_predictions \
    --num_images 10

if [ $? -ne 0 ]; then
    echo "❌ Ошибка при визуализации!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Quick Start завершен!"
echo "=========================================="
echo ""
echo "Результаты:"
echo "  - Модель: ./outputs/quick_test/checkpoints/"
echo "  - Метрики: ./outputs/quick_test/metrics.json"
echo "  - Анализ ошибок: ./visualizations/quick_test_errors/"
echo "  - Визуализации: ./visualizations/quick_test_predictions/"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir ./outputs/quick_test/logs"
echo ""
echo "Для полного обучения запустите:"
echo "  python src/train.py --num_epochs 10"
echo ""

