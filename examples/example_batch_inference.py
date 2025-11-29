"""
Пример: Batch инференс на множестве изображений
"""
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import sys
sys.path.append('..')

from src.dataset import CocoSubsetDataset, collate_fn, prepare_coco_subset
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor


def batch_inference(
    model,
    data_loader,
    class_names,
    device='cuda',
    threshold=0.5,
    save_dir=None
):
    """Batch инференс"""
    model.eval()
    
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing batches")):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels']
        
        # Инференс
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        
        # Обработка каждого изображения в батче
        for i in range(len(pixel_values)):
            image_id = labels[i]['image_id'].item()
            orig_h, orig_w = labels[i]['orig_size'].tolist()
            
            # Предсказания
            probas = outputs.logits[i].softmax(-1)[:, :-1]
            keep = probas.max(-1).values > threshold
            
            boxes = outputs.pred_boxes[i, keep].cpu()
            scores = probas[keep].max(-1).values.cpu()
            pred_labels = probas[keep].argmax(-1).cpu()
            
            # Конвертация боксов
            detections = []
            for box, score, label in zip(boxes, scores, pred_labels):
                cx, cy, bw, bh = box.numpy()
                
                # Денормализация
                x1 = (cx - bw/2) * orig_w
                y1 = (cy - bh/2) * orig_h
                x2 = (cx + bw/2) * orig_w
                y2 = (cy + bh/2) * orig_h
                
                detections.append({
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(score),
                    'class': class_names[label.item()],
                    'class_id': int(label)
                })
            
            all_results.append({
                'image_id': image_id,
                'detections': detections
            })
    
    # Сохранение результатов
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = save_dir / 'batch_inference_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nРезультаты сохранены: {results_path}")
        
        # Статистика
        total_detections = sum(len(r['detections']) for r in all_results)
        avg_detections = total_detections / len(all_results) if all_results else 0
        
        stats = {
            'total_images': len(all_results),
            'total_detections': total_detections,
            'avg_detections_per_image': avg_detections,
            'threshold': threshold
        }
        
        stats_path = save_dir / 'inference_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Статистика сохранена: {stats_path}")
        print(f"\nВсего изображений: {stats['total_images']}")
        print(f"Всего детекций: {stats['total_detections']}")
        print(f"Среднее детекций на изображение: {stats['avg_detections_per_image']:.2f}")
    
    return all_results


def main():
    """Пример использования"""
    
    # Параметры
    DATA_DIR = './data/coco'
    CHECKPOINT = './outputs/baseline/checkpoints/best_model.pt'
    CONFIG = './outputs/baseline/config.json'
    OUTPUT_DIR = './outputs/batch_inference'
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    THRESHOLD = 0.5
    MAX_SAMPLES = 100  # None для всех
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Устройство: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Загрузка конфига
    with open(CONFIG, 'r') as f:
        config = json.load(f)
    
    # Подготовка датасета
    print("\nПодготовка датасета...")
    dataset_info = prepare_coco_subset(DATA_DIR)
    dataset_info['selected_classes'] = config['selected_classes']
    
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    
    dataset = CocoSubsetDataset(
        img_folder=dataset_info['val_img_folder'],
        ann_file=dataset_info['val_ann_file'],
        processor=processor,
        selected_classes=dataset_info['selected_classes'],
        max_samples=MAX_SAMPLES
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )
    
    print(f"Количество изображений: {len(dataset)}")
    
    # Загрузка модели
    print("\nЗагрузка модели...")
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    
    # Batch инференс
    print("\nЗапуск batch inference...")
    results = batch_inference(
        model=model,
        data_loader=data_loader,
        class_names=config['selected_classes'],
        device=DEVICE,
        threshold=THRESHOLD,
        save_dir=OUTPUT_DIR
    )
    
    print(f"\n✅ Готово! Обработано {len(results)} изображений")


if __name__ == '__main__':
    main()

