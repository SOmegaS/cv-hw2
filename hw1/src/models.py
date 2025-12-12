"""
Модели: SimpleCNN и ViT-Tiny linear probe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging


class SimpleCNN(nn.Module):
    """
    Простая CNN архитектура с 3 свёрточными блоками
    
    Архитектура:
    - Conv Block 1: Conv2d(3->64) -> BatchNorm -> ReLU -> MaxPool
    - Conv Block 2: Conv2d(64->128) -> BatchNorm -> ReLU -> MaxPool  
    - Conv Block 3: Conv2d(128->256) -> BatchNorm -> ReLU -> MaxPool
    - Flatten -> FC(256*4*4->512) -> Dropout -> FC(512->num_classes)
    
    Примерно ~2.8M параметров
    """
    
    def __init__(self, num_classes=5, dropout=0.5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов с использованием kaiming_normal"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Block 1: 32x32 -> 16x16
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Block 2: 16x16 -> 8x8
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Block 3: 8x8 -> 4x4
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Flatten and FC
        x = x.view(x.size(0), -1)  # Flatten: [batch, 256*4*4]
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class ViTLinearProbe(nn.Module):
    """
    ViT-Tiny с замороженным backbone и обучаемым линейным классификатором
    
    Использует предобученный ViT-Tiny из библиотеки timm
    """
    
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=True):
        super(ViTLinearProbe, self).__init__()
        
        # Загружаем предобученный ViT-Tiny
        logging.info('Loading pretrained ViT-Tiny from timm...')
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
        
        # Получаем размерность выходных признаков
        num_features = self.vit.head.in_features
        
        # Замораживаем все слои backbone, если требуется
        if freeze_backbone:
            logging.info('Freezing ViT backbone...')
            for param in self.vit.parameters():
                param.requires_grad = False
        
        # Заменяем head на новый линейный классификатор
        self.vit.head = nn.Linear(num_features, num_classes)
        
        # Инициализируем новый head
        nn.init.normal_(self.vit.head.weight, std=0.01)
        nn.init.constant_(self.vit.head.bias, 0)
        
        logging.info(f'ViT-Tiny linear probe created with {num_features} features -> {num_classes} classes')
    
    def forward(self, x):
        return self.vit(x)
    
    def unfreeze_last_n_blocks(self, n=1):
        """
        Размораживает последние n блоков ViT для fine-tuning
        
        Args:
            n: количество последних блоков для разморозки
        """
        # Размораживаем последние n блоков
        total_blocks = len(self.vit.blocks)
        for i in range(total_blocks - n, total_blocks):
            for param in self.vit.blocks[i].parameters():
                param.requires_grad = True
        
        logging.info(f'Unfrozen last {n} blocks of ViT')


def create_model(model_type='cnn', num_classes=5, **kwargs):
    """
    Фабрика для создания моделей
    
    Args:
        model_type: 'cnn' или 'vit'
        num_classes: количество выходных классов
        **kwargs: дополнительные параметры для модели
        
    Returns:
        Модель PyTorch
    """
    if model_type.lower() == 'cnn':
        model = SimpleCNN(num_classes=num_classes, **kwargs)
        logging.info('Created SimpleCNN model')
    elif model_type.lower() == 'vit':
        model = ViTLinearProbe(num_classes=num_classes, **kwargs)
        logging.info('Created ViT-Tiny linear probe')
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'cnn' or 'vit'")
    
    return model


if __name__ == '__main__':
    # Тестирование моделей
    logging.basicConfig(level=logging.INFO)
    
    # Тест CNN
    cnn = SimpleCNN(num_classes=5)
    x_cnn = torch.randn(4, 3, 32, 32)
    out_cnn = cnn(x_cnn)
    print(f'CNN output shape: {out_cnn.shape}')
    print(f'CNN parameters: {sum(p.numel() for p in cnn.parameters() if p.requires_grad):,}')
    
    # Тест ViT
    vit = ViTLinearProbe(num_classes=5, pretrained=False)
    x_vit = torch.randn(4, 3, 224, 224)
    out_vit = vit(x_vit)
    print(f'ViT output shape: {out_vit.shape}')
    print(f'ViT trainable parameters: {sum(p.numel() for p in vit.parameters() if p.requires_grad):,}')
    print(f'ViT total parameters: {sum(p.numel() for p in vit.parameters()):,}')
