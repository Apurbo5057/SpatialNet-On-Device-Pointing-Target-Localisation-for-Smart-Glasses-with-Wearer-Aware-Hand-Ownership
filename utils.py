"""
Export FP32, FP16 and dynamic-INT8 copies of a trained SpatialNet.
Called by train.py --quantize.
"""

import os
import warnings
import torch
import torch.nn as nn


def save_quantized_models(model, output_dir):
    """Save model_fp32/fp16/int8.pth; returns {format: {'path', 'size_kb'}}.
    Dynamic INT8 quantization only affects nn.Linear; conv layers stay FP32."""
    os.makedirs(output_dir, exist_ok=True)
    model_cpu = model.cpu().eval()
    results = {}

    fp32_path = os.path.join(output_dir, 'model_fp32.pth')
    torch.save(model_cpu.state_dict(), fp32_path)
    results['fp32'] = {'path': fp32_path, 'size_kb': os.path.getsize(fp32_path) / 1024}

    fp16_path = os.path.join(output_dir, 'model_fp16.pth')
    torch.save({k: (v.half() if v.is_floating_point() else v)
                for k, v in model_cpu.state_dict().items()}, fp16_path)
    results['fp16'] = {'path': fp16_path, 'size_kb': os.path.getsize(fp16_path) / 1024}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_int8 = torch.quantization.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8)
    int8_path = os.path.join(output_dir, 'model_int8.pth')
    torch.save(model_int8.state_dict(), int8_path)
    results['int8'] = {'path': int8_path, 'size_kb': os.path.getsize(int8_path) / 1024}
    return results
