#!/usr/bin/env python3
"""
Скрипт для профилировки CNN и ViT-Tiny моделей
"""

import sys
from pathlib import Path as PathLib

# Add parent directory to path
sys.path.insert(0, str(PathLib(__file__).parent.parent))

import torch
import argparse
import logging
from pathlib import Path

from src.utils import set_seed, setup_logging, get_device
from src.data import get_cifar10_dataloaders, CLASS_NAMES
from src.models import SimpleCNN, ViTLinearProbe
from src.profiling import comprehensive_profiling, save_profiling_comparison


def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Profile CNN and ViT models')
    
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory for dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory with model checkpoints')
    parser.add_argument('--trace_dir', type=str, default='runs/profiler',
                       help='Directory for profiler traces')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory for saving results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size_cnn', type=int, default=128,
                       help='Batch size for CNN')
    parser.add_argument('--batch_size_vit', type=int, default=64,
                       help='Batch size for ViT')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    setup_logging()
    logging.info('='*60)
    logging.info('Profiling CNN and ViT-Tiny models')
    logging.info('='*60)
    
    set_seed(args.seed)
    device = get_device()
    
    results_list = []
    
    # ========== Профилировка CNN ==========
    logging.info('\n' + '='*60)
    logging.info('Profiling CNN')
    logging.info('='*60 + '\n')
    
    # Загружаем данные для CNN
    _, _, test_loader_cnn = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size_cnn,
        input_size=32,
        use_imagenet_stats=False,
        num_workers=2,
        seed=args.seed
    )
    
    # Создаем модель CNN
    cnn_model = SimpleCNN(num_classes=len(CLASS_NAMES))
    cnn_model = cnn_model.to(device)
    
    # Загружаем веса, если есть
    cnn_checkpoint_path = Path(args.checkpoint_dir) / 'cnn_best.pth'
    if cnn_checkpoint_path.exists():
        checkpoint = torch.load(cnn_checkpoint_path, map_location=device)
        cnn_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Loaded CNN weights from {cnn_checkpoint_path}')
    else:
        logging.warning(f'CNN checkpoint not found at {cnn_checkpoint_path}, using random weights')
    
    cnn_model.eval()
    
    # Профилировка CNN
    cnn_results = comprehensive_profiling(
        model=cnn_model,
        data_loader=test_loader_cnn,
        device=device,
        model_name='cnn',
        input_shape=(args.batch_size_cnn, 3, 32, 32),
        trace_dir=args.trace_dir
    )
    results_list.append(cnn_results)
    
    # ========== Профилировка ViT ==========
    logging.info('\n' + '='*60)
    logging.info('Profiling ViT-Tiny')
    logging.info('='*60 + '\n')
    
    # Загружаем данные для ViT
    _, _, test_loader_vit = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size_vit,
        input_size=224,
        use_imagenet_stats=True,
        num_workers=2,
        seed=args.seed
    )
    
    # Создаем модель ViT
    vit_model = ViTLinearProbe(num_classes=len(CLASS_NAMES), pretrained=True, freeze_backbone=True)
    vit_model = vit_model.to(device)
    
    # Загружаем веса, если есть
    vit_checkpoint_path = Path(args.checkpoint_dir) / 'vit_best.pth'
    if vit_checkpoint_path.exists():
        checkpoint = torch.load(vit_checkpoint_path, map_location=device)
        vit_model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f'Loaded ViT weights from {vit_checkpoint_path}')
    else:
        logging.warning(f'ViT checkpoint not found at {vit_checkpoint_path}, using default weights')
    
    vit_model.eval()
    
    # Профилировка ViT
    vit_results = comprehensive_profiling(
        model=vit_model,
        data_loader=test_loader_vit,
        device=device,
        model_name='vit',
        input_shape=(args.batch_size_vit, 3, 224, 224),
        trace_dir=args.trace_dir
    )
    results_list.append(vit_results)
    
    # ========== Сохранение сравнения ==========
    logging.info('\n' + '='*60)
    logging.info('Saving profiling comparison')
    logging.info('='*60)
    
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    save_profiling_comparison(results_list, f'{args.results_dir}/profiling_comparison')
    
    logging.info('\n' + '='*60)
    logging.info('Profiling completed!')
    logging.info('='*60)
    
    # Выводим сводку
    logging.info('\nComparison Summary:')
    logging.info('-'*60)
    for result in results_list:
        logging.info(f"\n{result['model_name'].upper()}:")
        logging.info(f"  Total params: {result['total_params']:,}")
        logging.info(f"  Trainable params: {result['trainable_params']:,}")
        logging.info(f"  Throughput: {result['throughput_images_per_sec']:.2f} img/sec")
        logging.info(f"  Latency: {result['latency_ms_per_batch']:.2f} ms/batch")
        if device.type == 'cuda':
            logging.info(f"  Peak memory: {result['peak_memory_mb']:.2f} MB")


if __name__ == '__main__':
    main()
