"""
Тренировочный цикл с TensorBoard логированием и sanity checks
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm


class Trainer:
    """
    Класс для обучения моделей с TensorBoard логированием
    """
    
    def __init__(self, model, device, train_loader, val_loader, 
                 criterion, optimizer, scheduler=None,
                 log_dir='runs', checkpoint_dir='checkpoints',
                 model_name='model'):
        """
        Args:
            model: PyTorch модель
            device: устройство для обучения
            train_loader: DataLoader для train
            val_loader: DataLoader для validation
            criterion: функция потерь
            optimizer: оптимизатор
            scheduler: learning rate scheduler (опционально)
            log_dir: директория для TensorBoard логов
            checkpoint_dir: директория для чекпоинтов
            model_name: имя модели для логирования
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        
        # Создаем директории
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=f'{log_dir}/{model_name}')
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.current_epoch = 0
        
    def train_epoch(self):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Обновляем progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': running_loss / len(self.val_loader),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def log_to_tensorboard(self, train_loss, train_acc, val_loss, val_acc):
        """Логирование метрик в TensorBoard"""
        epoch = self.current_epoch
        
        # Loss и accuracy
        self.writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        
        # Learning rate
        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Гистограммы весов и градиентов (каждые 5 эпох)
        if epoch % 5 == 0:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'Weights/{name}', param.data, epoch)
                    if param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
    
    def save_checkpoint(self, filename='best_model.pth'):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = Path(self.checkpoint_dir) / filename
        torch.save(checkpoint, save_path)
        logging.info(f'Checkpoint saved to {save_path}')
    
    def load_checkpoint(self, filename='best_model.pth'):
        """Загрузка чекпоинта"""
        load_path = Path(self.checkpoint_dir) / filename
        if not load_path.exists():
            logging.warning(f'Checkpoint {load_path} not found')
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accs = checkpoint.get('train_accs', [])
        self.val_accs = checkpoint.get('val_accs', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        
        logging.info(f'Checkpoint loaded from {load_path}')
    
    def train(self, num_epochs, early_stopping_patience=10, save_best_only=True):
        """
        Основной цикл обучения
        
        Args:
            num_epochs: количество эпох
            early_stopping_patience: количество эпох без улучшения для early stopping
            save_best_only: сохранять только лучшую модель
        """
        logging.info(f'Starting training for {num_epochs} epochs')
        logging.info(f'Device: {self.device}')
        logging.info(f'Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}')
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Сохраняем историю
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Логируем в TensorBoard
            self.log_to_tensorboard(train_loss, train_acc, val_loss, val_acc)
            
            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Логирование
            logging.info(
                f'Epoch {epoch + 1}/{num_epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            
            # Сохранение лучшей модели
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                if save_best_only:
                    self.save_checkpoint(f'{self.model_name}_best.pth')
                    logging.info(f'New best model! Val Acc: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logging.info(f'Early stopping triggered after {epoch + 1} epochs')
                break
        
        total_time = time.time() - start_time
        logging.info(f'Training completed in {total_time:.2f}s')
        logging.info(f'Best Val Acc: {self.best_val_acc:.2f}%, Best Val Loss: {self.best_val_loss:.4f}')
        
        self.writer.close()
        
        return self.best_val_acc, self.best_val_loss


def sanity_check_overfit(model, device, train_loader, num_epochs=50):
    """
    Sanity check: проверка способности модели переобучиться на маленьком батче
    
    Args:
        model: модель для проверки
        device: устройство
        train_loader: DataLoader с маленьким батчом
        num_epochs: количество эпох для overfitting
        
    Returns:
        True если модель достигла >95% accuracy, иначе False
    """
    logging.info('Running sanity check: overfitting on small batch...')
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Берем один батч
    inputs, targets = next(iter(train_loader))
    inputs, targets = inputs.to(device), targets.to(device)
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        acc = 100. * predicted.eq(targets).sum().item() / targets.size(0)
        
        if (epoch + 1) % 10 == 0:
            logging.info(f'Sanity Check Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}, Acc: {acc:.2f}%')
        
        # Если достигли высокой точности, проверка пройдена
        if acc > 95.0:
            logging.info(f'✓ Sanity check PASSED! Achieved {acc:.2f}% accuracy at epoch {epoch + 1}')
            return True
    
    logging.warning(f'✗ Sanity check FAILED! Only achieved {acc:.2f}% accuracy')
    return False
