"""
Оценка моделей и вычисление метрик
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_recall_fscore_support
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import json


def evaluate_model(model, test_loader, device, class_names):
    """
    Оценка модели на тестовой выборке
    
    Args:
        model: PyTorch модель
        test_loader: DataLoader для test
        device: устройство для вычислений
        class_names: список имен классов
        
    Returns:
        Dict с метриками и предсказаниями
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    logging.info('Evaluating model on test set...')
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Вычисляем метрики
    accuracy = 100. * np.sum(all_preds == all_targets) / len(all_targets)
    
    # Macro F1 score
    macro_f1 = f1_score(all_targets, all_preds, average='macro') * 100
    
    # Per-class метрики
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, average=None, labels=range(len(class_names))
    )
    
    # Classification report
    report = classification_report(
        all_targets, all_preds, 
        target_names=class_names,
        digits=4
    )
    
    results = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1,
        'support': support,
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs,
        'classification_report': report
    }
    
    logging.info(f'Test Accuracy: {accuracy:.2f}%')
    logging.info(f'Macro F1-Score: {macro_f1:.2f}%')
    logging.info('\nClassification Report:')
    logging.info('\n' + report)
    
    return results


def plot_confusion_matrix(targets, predictions, class_names, save_path):
    """
    Построение и сохранение confusion matrix
    
    Args:
        targets: истинные метки
        predictions: предсказания модели
        class_names: список имен классов
        save_path: путь для сохранения изображения
    """
    cm = confusion_matrix(targets, predictions)
    
    # Нормализуем по строкам (истинным классам)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Абсолютные значения
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (Absolute)')
    
    # Нормализованные значения
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Confusion matrix saved to {save_path}')


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """
    Визуализация истории обучения
    
    Args:
        train_losses: список train losses
        val_losses: список val losses
        train_accs: список train accuracies
        val_accs: список val accuracies
        save_path: путь для сохранения
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f'Training history plot saved to {save_path}')


def save_metrics_table(metrics_dict, save_path):
    """
    Сохранение таблицы метрик в CSV и Markdown
    
    Args:
        metrics_dict: словарь с метриками разных моделей
        save_path: базовый путь для сохранения (без расширения)
    """
    df = pd.DataFrame(metrics_dict).T
    
    # Сохраняем в CSV
    csv_path = f'{save_path}.csv'
    df.to_csv(csv_path)
    logging.info(f'Metrics table saved to {csv_path}')
    
    # Сохраняем в Markdown
    md_path = f'{save_path}.md'
    with open(md_path, 'w') as f:
        f.write(df.to_markdown())
    logging.info(f'Metrics table saved to {md_path}')
    
    return df


def save_detailed_results(results, model_name, class_names, save_dir):
    """
    Сохранение детальных результатов оценки модели
    
    Args:
        results: результаты из evaluate_model
        model_name: имя модели
        class_names: список имен классов
        save_dir: директория для сохранения
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем confusion matrix
    cm_path = save_dir / f'confusion_matrix_{model_name}.png'
    plot_confusion_matrix(
        results['targets'], 
        results['predictions'],
        class_names,
        cm_path
    )
    
    # Сохраняем classification report
    report_path = save_dir / f'classification_report_{model_name}.txt'
    with open(report_path, 'w') as f:
        f.write(results['classification_report'])
    logging.info(f'Classification report saved to {report_path}')
    
    # Сохраняем per-class метрики
    per_class_metrics = {
        'class': class_names,
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1_per_class'],
        'support': results['support']
    }
    df_per_class = pd.DataFrame(per_class_metrics)
    per_class_path = save_dir / f'per_class_metrics_{model_name}.csv'
    df_per_class.to_csv(per_class_path, index=False)
    logging.info(f'Per-class metrics saved to {per_class_path}')
    
    # Сохраняем общие метрики
    summary_metrics = {
        'accuracy': results['accuracy'],
        'macro_f1': results['macro_f1']
    }
    summary_path = save_dir / f'summary_metrics_{model_name}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    logging.info(f'Summary metrics saved to {summary_path}')


def compare_models(results_dict, class_names, save_dir='results'):
    """
    Сравнение нескольких моделей
    
    Args:
        results_dict: словарь {model_name: results}
        class_names: список имен классов
        save_dir: директория для сохранения
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем сводную таблицу
    comparison = {}
    for model_name, results in results_dict.items():
        comparison[model_name] = {
            'Accuracy (%)': f"{results['accuracy']:.2f}",
            'Macro F1 (%)': f"{results['macro_f1']:.2f}",
        }
        
        # Добавляем per-class F1
        for i, class_name in enumerate(class_names):
            comparison[model_name][f'F1-{class_name} (%)'] = f"{results['f1_per_class'][i] * 100:.2f}"
    
    # Сохраняем таблицу сравнения
    save_metrics_table(comparison, save_dir / 'model_comparison')
    
    logging.info('Model comparison completed')
