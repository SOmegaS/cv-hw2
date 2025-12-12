"""
Загрузка и подготовка данных CIFAR-10 (subset из 5 классов)
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import logging


# Выбираем 5 классов из CIFAR-10
SELECTED_CLASSES = [0, 1, 2, 3, 4]  # airplane, automobile, bird, cat, dog
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'dog']

# Статистика ImageNet для нормализации (используется для ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Статистика CIFAR-10
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2470, 0.2435, 0.2616]


def get_cifar10_transforms(input_size=32, use_imagenet_stats=False):
    """
    Создает трансформации для train и val наборов
    
    Args:
        input_size: размер входного изображения (32 для CNN, 224 для ViT)
        use_imagenet_stats: использовать статистику ImageNet (для ViT)
        
    Returns:
        Tuple (train_transform, val_transform)
    """
    mean = IMAGENET_MEAN if use_imagenet_stats else CIFAR10_MEAN
    std = IMAGENET_STD if use_imagenet_stats else CIFAR10_STD
    
    # Аугментации для train
    train_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Без аугментаций для val
    val_transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform


def filter_dataset_by_classes(dataset, selected_classes):
    """
    Фильтрует датасет, оставляя только выбранные классы
    
    Args:
        dataset: PyTorch Dataset
        selected_classes: список индексов классов для отбора
        
    Returns:
        Subset с отфильтрованными данными
    """
    indices = [i for i, (_, label) in enumerate(dataset) if label in selected_classes]
    return Subset(dataset, indices)


def remap_labels(dataset, selected_classes):
    """
    Переназначает метки классов на диапазон [0, num_classes-1]
    
    Args:
        dataset: Dataset или Subset
        selected_classes: список исходных индексов классов
        
    Returns:
        Dataset с переназначенными метками
    """
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, class_mapping):
            self.dataset = dataset
            self.class_mapping = class_mapping
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            img, label = self.dataset[idx]
            return img, self.class_mapping[label]
    
    class_mapping = {old_label: new_label for new_label, old_label in enumerate(selected_classes)}
    return RemappedDataset(dataset, class_mapping)


def get_cifar10_dataloaders(data_dir='./data', 
                            batch_size=128,
                            val_split=0.2,
                            input_size=32,
                            use_imagenet_stats=False,
                            num_workers=4,
                            seed=42):
    """
    Создает DataLoader'ы для train, val и test наборов CIFAR-10 (5 классов)
    
    Args:
        data_dir: директория для сохранения данных
        batch_size: размер батча
        val_split: доля валидационной выборки от train
        input_size: размер изображений (32 для CNN, 224 для ViT)
        use_imagenet_stats: использовать ImageNet статистику для нормализации
        num_workers: количество worker'ов для загрузки данных
        seed: random seed для воспроизводимости
        
    Returns:
        Tuple (train_loader, val_loader, test_loader)
    """
    train_transform, val_transform = get_cifar10_transforms(input_size, use_imagenet_stats)
    
    # Загружаем полный CIFAR-10
    logging.info('Downloading CIFAR-10 dataset...')
    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                          download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                    download=True, transform=val_transform)
    
    # Фильтруем по выбранным классам
    logging.info(f'Filtering dataset to classes: {CLASS_NAMES}')
    train_dataset = filter_dataset_by_classes(full_train_dataset, SELECTED_CLASSES)
    test_dataset = filter_dataset_by_classes(test_dataset, SELECTED_CLASSES)
    
    # Переназначаем метки классов на [0, 4]
    train_dataset = remap_labels(train_dataset, SELECTED_CLASSES)
    test_dataset = remap_labels(test_dataset, SELECTED_CLASSES)
    
    # Разделяем train на train и val
    train_size = int((1 - val_split) * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], 
                                              generator=generator)
    
    logging.info(f'Dataset sizes - Train: {len(train_dataset)}, '
                f'Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


def sanity_check_overfit_batch(model, device, num_samples=32, input_size=32, 
                               use_imagenet_stats=False):
    """
    Создает маленький батч для sanity check (проверка overfitting)
    
    Args:
        model: модель для проверки
        device: устройство для вычислений
        num_samples: количество сэмплов для overfitting
        input_size: размер входных изображений
        use_imagenet_stats: использовать ImageNet статистику
        
    Returns:
        DataLoader с маленьким батчом
    """
    train_transform, _ = get_cifar10_transforms(input_size, use_imagenet_stats)
    
    full_dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=True, transform=train_transform)
    filtered_dataset = filter_dataset_by_classes(full_dataset, SELECTED_CLASSES)
    remapped_dataset = remap_labels(filtered_dataset, SELECTED_CLASSES)
    
    # Берем только первые num_samples сэмплов
    subset = Subset(remapped_dataset, list(range(num_samples)))
    
    return DataLoader(subset, batch_size=num_samples, shuffle=True)
