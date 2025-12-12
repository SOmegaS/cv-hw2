"""
Вспомогательные функции для обучения и воспроизводимости
"""

import random
import numpy as np
import torch
import logging


def set_seed(seed=42):
    """
    Фиксирует все random seeds для воспроизводимости результатов
    
    Args:
        seed: значение seed для генераторов случайных чисел
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def setup_logging(log_file=None):
    """
    Настраивает логирование для вывода в консоль и опционально в файл
    
    Args:
        log_file: путь к файлу для сохранения логов (опционально)
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def count_parameters(model):
    """
    Подсчитывает количество обучаемых параметров в модели
    
    Args:
        model: PyTorch модель
        
    Returns:
        Количество обучаемых параметров
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device():
    """
    Определяет доступное устройство (CUDA/CPU)
    
    Returns:
        torch.device
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f'Using CUDA: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        logging.info('Using CPU')
    return device
