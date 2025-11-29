#!/bin/bash

# Пример: различные сценарии обучения

# Активация окружения
source ../venv/bin/activate

echo "Примеры запуска обучения DETR"
echo ""

# ============================================
# Пример 1: Быстрый тест (малая выборка)
# ============================================
echo "1. Быстрый тест (50 train, 20 val samples, 2 epochs)"
echo "----------------------------------------"
cat << 'EOF'
python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/test \
    --batch_size 2 \
    --num_epochs 2 \
    --max_train_samples 50 \
    --max_val_samples 20
EOF
echo ""

# ============================================
# Пример 2: Стандартное обучение
# ============================================
echo "2. Стандартное обучение (10 epochs)"
echo "----------------------------------------"
cat << 'EOF'
python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/baseline \
    --model_name facebook/detr-resnet-50 \
    --batch_size 4 \
    --num_epochs 10 \
    --lr 1e-5 \
    --num_workers 4 \
    --profile_epoch 2
EOF
echo ""

# ============================================
# Пример 3: Deformable DETR (более точная модель)
# ============================================
echo "3. Deformable DETR"
echo "----------------------------------------"
cat << 'EOF'
python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/deformable \
    --model_name SenseTime/deformable-detr \
    --batch_size 2 \
    --num_epochs 15 \
    --lr 2e-5
EOF
echo ""

# ============================================
# Пример 4: Обучение на GPU с большим батчем
# ============================================
echo "4. Большой батч (требуется GPU с 16GB+)"
echo "----------------------------------------"
cat << 'EOF'
python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/large_batch \
    --batch_size 8 \
    --num_epochs 10 \
    --lr 1e-5 \
    --num_workers 8
EOF
echo ""

# ============================================
# Пример 5: Fine-tuning с низкой скоростью обучения
# ============================================
echo "5. Fine-tuning (медленное обучение)"
echo "----------------------------------------"
cat << 'EOF'
python src/train.py \
    --data_dir ./data/coco \
    --output_dir ./outputs/finetune \
    --batch_size 4 \
    --num_epochs 20 \
    --lr 5e-6
EOF
echo ""

echo "Выберите и запустите нужный пример, удалив 'cat << EOF ... EOF' и раскомментировав команду"

