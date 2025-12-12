"""
Профилировка моделей с использованием torch.profiler
"""

import torch
import torch.profiler
import logging
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def profile_model(model, data_loader, device, num_steps=100, 
                 trace_dir='runs/profiler', model_name='model'):
    """
    Профилировка модели с использованием torch.profiler
    
    Args:
        model: PyTorch модель
        data_loader: DataLoader для профилировки
        device: устройство для вычислений
        num_steps: количество шагов для профилировки
        trace_dir: директория для сохранения trace
        model_name: имя модели
        
    Returns:
        Dict с метриками производительности
    """
    model.eval()
    
    trace_path = Path(trace_dir) / model_name
    trace_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f'Starting profiling for {model_name}...')
    
    # Подготовка данных
    data_iter = iter(data_loader)
    
    def trace_handler(prof):
        """Handler для сохранения trace"""
        output = prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20)
        logging.info(f'\n{output}')
        prof.export_chrome_trace(str(trace_path / f'{model_name}_trace.json'))
    
    # Настройка профайлера
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=2,
            warmup=2,
            active=6,
            repeat=1
        ),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        
        with torch.no_grad():
            for step in range(min(num_steps, len(data_loader))):
                try:
                    inputs, _ = next(data_iter)
                except StopIteration:
                    data_iter = iter(data_loader)
                    inputs, _ = next(data_iter)
                
                inputs = inputs.to(device)
                _ = model(inputs)
                prof.step()
    
    logging.info(f'Profiling completed. Trace saved to {trace_path}/{model_name}_trace.json')
    
    # Профилировка завершена, trace сохранен
    return {
        'trace_path': str(trace_path / f'{model_name}_trace.json')
    }


def measure_throughput(model, data_loader, device, num_batches=50, warmup_batches=5):
    """
    Измерение throughput модели (images/sec)
    
    Args:
        model: PyTorch модель
        data_loader: DataLoader
        device: устройство
        num_batches: количество батчей для измерения
        warmup_batches: количество warmup батчей
        
    Returns:
        Dict с метриками throughput
    """
    model.eval()
    
    logging.info(f'Measuring throughput...')
    
    data_iter = iter(data_loader)
    batch_size = data_loader.batch_size
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_batches):
            try:
                inputs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, _ = next(data_iter)
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Синхронизация для CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Измерение времени
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                inputs, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                inputs, _ = next(data_iter)
            inputs = inputs.to(device)
            _ = model(inputs)
    
    # Синхронизация для CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    total_images = num_batches * batch_size
    throughput = total_images / total_time
    latency = (total_time / num_batches) * 1000  # ms per batch
    
    logging.info(f'Throughput: {throughput:.2f} images/sec')
    logging.info(f'Latency: {latency:.2f} ms/batch')
    
    return {
        'throughput_images_per_sec': throughput,
        'latency_ms_per_batch': latency,
        'total_time_sec': total_time,
        'total_images': total_images
    }


def measure_memory_usage(model, input_shape, device):
    """
    Измерение использования памяти моделью
    
    Args:
        model: PyTorch модель
        input_shape: форма входного тензора (batch_size, channels, height, width)
        device: устройство
        
    Returns:
        Dict с метриками памяти
    """
    model.eval()
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        
        dummy_input = torch.randn(input_shape).to(device)
        
        # Измеряем память до forward pass
        memory_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Измеряем память после forward pass
        memory_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        del dummy_input
        torch.cuda.empty_cache()
        
        return {
            'memory_before_mb': memory_before,
            'memory_after_mb': memory_after,
            'peak_memory_mb': peak_memory,
            'memory_used_mb': memory_after - memory_before
        }
    else:
        logging.warning('Memory measurement is only available for CUDA devices')
        return {
            'memory_before_mb': 0,
            'memory_after_mb': 0,
            'peak_memory_mb': 0,
            'memory_used_mb': 0
        }


def count_flops(model, input_shape):
    """
    Подсчет FLOPs (приблизительный)
    
    Args:
        model: PyTorch модель
        input_shape: форма входного тензора
        
    Returns:
        Количество FLOPs
    """
    # Простой подсчет на основе параметров
    # Для более точного подсчета можно использовать библиотеки типа ptflops или fvcore
    total_params = sum(p.numel() for p in model.parameters())
    
    # Грубая оценка: ~2 * params (умножение и сложение)
    approx_flops = 2 * total_params
    
    return {
        'approx_flops': approx_flops,
        'total_params': total_params
    }


def comprehensive_profiling(model, data_loader, device, model_name, 
                           input_shape, trace_dir='runs/profiler'):
    """
    Комплексная профилировка модели
    
    Args:
        model: PyTorch модель
        data_loader: DataLoader
        device: устройство
        model_name: имя модели
        input_shape: форма входного тензора
        trace_dir: директория для trace
        
    Returns:
        Dict со всеми метриками
    """
    logging.info(f'\n{"="*60}')
    logging.info(f'Comprehensive profiling for {model_name}')
    logging.info(f'{"="*60}\n')
    
    # Профилировка с torch.profiler
    profile_results = profile_model(model, data_loader, device, 
                                   num_steps=100, trace_dir=trace_dir, 
                                   model_name=model_name)
    
    # Измерение throughput
    throughput_results = measure_throughput(model, data_loader, device)
    
    # Измерение памяти
    memory_results = measure_memory_usage(model, input_shape, device)
    
    # Подсчет FLOPs
    flops_results = count_flops(model, input_shape)
    
    # Количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Объединяем все результаты
    results = {
        'model_name': model_name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        **profile_results,
        **throughput_results,
        **memory_results,
        **flops_results
    }
    
    # Логируем сводку
    logging.info(f'\nProfiling Summary for {model_name}:')
    logging.info(f'Total Parameters: {total_params:,}')
    logging.info(f'Trainable Parameters: {trainable_params:,}')
    logging.info(f'Throughput: {throughput_results["throughput_images_per_sec"]:.2f} images/sec')
    logging.info(f'Latency: {throughput_results["latency_ms_per_batch"]:.2f} ms/batch')
    if device.type == 'cuda':
        logging.info(f'Peak Memory: {memory_results["peak_memory_mb"]:.2f} MB')
    
    return results


def save_profiling_comparison(results_list, save_path='results/profiling_comparison'):
    """
    Сохранение сравнения профилировки нескольких моделей
    
    Args:
        results_list: список словарей с результатами профилировки
        save_path: путь для сохранения (без расширения)
    """
    df = pd.DataFrame(results_list)
    
    # Выбираем важные колонки для отображения
    display_columns = [
        'model_name', 
        'total_params', 
        'trainable_params',
        'throughput_images_per_sec',
        'latency_ms_per_batch',
        'peak_memory_mb'
    ]
    
    df_display = df[display_columns]
    
    # Сохраняем в CSV
    csv_path = f'{save_path}.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f'Profiling comparison saved to {csv_path}')
    
    # Сохраняем в Markdown
    md_path = f'{save_path}.md'
    with open(md_path, 'w') as f:
        f.write(df_display.to_markdown(index=False))
    logging.info(f'Profiling comparison saved to {md_path}')
    
    return df
