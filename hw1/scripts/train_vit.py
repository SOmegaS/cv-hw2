#!/usr/bin/env python3
"""
Скрипт для обучения ViT-Tiny linear probe
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
from pathlib import Path

from src.utils import set_seed, setup_logging, get_device, count_parameters
from src.data import get_cifar10_dataloaders, sanity_check_overfit_batch, CLASS_NAMES
from src.models import ViTLinearProbe
from src.train import Trainer, sanity_check_overfit


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Train ViT-Tiny linear probe on CIFAR-10 subset')
    
    # Параметры данных
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--batch_size', type=int, default=64,  # Меньше из-за большего размера изображений
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Параметры модели
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained ViT-Tiny')
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze backbone layers')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=50,  # Меньше эпох, т.к. linear probe быстро сходится
                       help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01,  # Выше LR для linear probe
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Другие параметры
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='runs',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory for model checkpoints')
    parser.add_argument('--skip_sanity_check', action='store_true',
                       help='Skip sanity check (overfit test)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Настройка логирования
    setup_logging()
    logging.info('='*60)
    logging.info('Training ViT-Tiny linear probe on CIFAR-10 subset')
    logging.info('='*60)
    
    # Фиксируем seed
    set_seed(args.seed)
    logging.info(f'Random seed set to {args.seed}')
    
    # Устройство
    device = get_device()
    
    # Загружаем данные (для ViT нужно 224x224 и ImageNet статистика)
    logging.info('Loading CIFAR-10 dataset...')
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        input_size=224,  # ViT требует 224x224
        use_imagenet_stats=True,  # Используем ImageNet статистику
        num_workers=args.num_workers,
        seed=args.seed
    )
    
    # Создаем модель
    logging.info('Creating ViT-Tiny linear probe...')
    model = ViTLinearProbe(
        num_classes=len(CLASS_NAMES),
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)
    
    trainable_params = count_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Model has {trainable_params:,} trainable parameters out of {total_params:,} total')
    
    # Sanity check
    if not args.skip_sanity_check:
        logging.info('\n' + '='*60)
        logging.info('Running sanity check: overfitting on small batch')
        logging.info('='*60)
        
        sanity_loader = sanity_check_overfit_batch(
            model, device, num_samples=32, input_size=224, use_imagenet_stats=True
        )
        
        # Создаем копию модели для sanity check
        sanity_model = ViTLinearProbe(
            num_classes=len(CLASS_NAMES),
            pretrained=args.pretrained,
            freeze_backbone=args.freeze_backbone
        )
        sanity_model = sanity_model.to(device)
        
        passed = sanity_check_overfit(sanity_model, device, sanity_loader, num_epochs=30)
        
        if not passed:
            logging.warning('Sanity check did not pass completely, but continuing training...')
        
        del sanity_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logging.info('='*60 + '\n')
    
    # Loss и optimizer (оптимизируем только trainable параметры)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Создаем trainer
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        model_name='vit'
    )
    
    # Обучение
    logging.info('\n' + '='*60)
    logging.info('Starting training')
    logging.info('='*60 + '\n')
    
    best_val_acc, best_val_loss = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        save_best_only=True
    )
    
    # Финальная оценка на test set
    logging.info('\n' + '='*60)
    logging.info('Evaluating on test set')
    logging.info('='*60)
    
    # Загружаем лучшую модель
    trainer.load_checkpoint('vit_best.pth')
    
    # Оценка
    from src.evaluate import evaluate_model, save_detailed_results
    
    results = evaluate_model(model, test_loader, device, CLASS_NAMES)
    save_detailed_results(results, 'vit', CLASS_NAMES, 'results')
    
    logging.info('\n' + '='*60)
    logging.info('Training completed!')
    logging.info(f'Best validation accuracy: {best_val_acc:.2f}%')
    logging.info(f'Test accuracy: {results["accuracy"]:.2f}%')
    logging.info(f'Test macro F1: {results["macro_f1"]:.2f}%')
    logging.info('='*60)


if __name__ == '__main__':
    main()
