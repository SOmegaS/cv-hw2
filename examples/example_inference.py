"""
Пример: Инференс на одном изображении
"""
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrForObjectDetection, DetrImageProcessor
import json
from pathlib import Path


def load_model(checkpoint_path, config_path, device='cuda'):
    """Загрузка модели из чекпойнта"""
    # Загрузка конфига
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Инициализация модели
    model = DetrForObjectDetection.from_pretrained(
        config['model_name'],
        num_labels=config['num_classes'],
        ignore_mismatched_sizes=True
    )
    
    # Загрузка весов
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Процессор
    processor = DetrImageProcessor.from_pretrained(config['model_name'])
    
    return model, processor, config['selected_classes']


def detect_objects(image_path, model, processor, class_names, device='cuda', threshold=0.5):
    """Детекция объектов на изображении"""
    # Загрузка изображения
    image = Image.open(image_path).convert('RGB')
    
    # Препроцессинг
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Инференс
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
    
    # Обработка предсказаний
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    
    boxes = outputs.pred_boxes[0, keep].cpu()
    scores = probas[keep].max(-1).values.cpu()
    labels = probas[keep].argmax(-1).cpu()
    
    # Конвертация в формат [x1, y1, x2, y2]
    w, h = image.size
    boxes_scaled = []
    for box in boxes:
        cx, cy, bw, bh = box.numpy()
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        x2 = (cx + bw/2) * w
        y2 = (cy + bh/2) * h
        boxes_scaled.append([x1, y1, x2, y2])
    
    results = []
    for box, score, label in zip(boxes_scaled, scores, labels):
        results.append({
            'box': box,
            'score': score.item(),
            'label': class_names[label.item()]
        })
    
    return image, results


def visualize_detections(image, results, save_path=None):
    """Визуализация результатов"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Отрисовка боксов
    for result in results:
        x1, y1, x2, y2 = result['box']
        width = x2 - x1
        height = y2 - y1
        
        # Бокс
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Метка
        label_text = f"{result['label']}: {result['score']:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.8)
        )
    
    ax.axis('off')
    plt.title(f'Detected {len(results)} objects', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Пример использования"""
    
    # Параметры
    CHECKPOINT = './outputs/baseline/checkpoints/best_model.pt'
    CONFIG = './outputs/baseline/config.json'
    IMAGE_PATH = './data/coco/val2017/000000000139.jpg'  # Замените на свое изображение
    OUTPUT_PATH = './outputs/detection_result.jpg'
    THRESHOLD = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Устройство: {DEVICE}")
    print(f"Изображение: {IMAGE_PATH}")
    
    # Загрузка модели
    print("Загрузка модели...")
    model, processor, class_names = load_model(CHECKPOINT, CONFIG, DEVICE)
    print(f"Классы: {class_names}")
    
    # Детекция
    print("Детекция объектов...")
    image, results = detect_objects(
        IMAGE_PATH, model, processor, class_names, 
        device=DEVICE, threshold=THRESHOLD
    )
    
    # Вывод результатов
    print(f"\nНайдено объектов: {len(results)}")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['label']}: {result['score']:.3f} - box: {[f'{x:.1f}' for x in result['box']]}")
    
    # Визуализация
    print("\nВизуализация...")
    visualize_detections(image, results, save_path=OUTPUT_PATH)
    
    print(f"\n✅ Готово! Результат сохранен: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()

