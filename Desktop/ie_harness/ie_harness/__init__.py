"""IE-CRSI-Harness: 图像增强专属的 Harness 自进化框架。"""
import torch

# CPU 推理固定线程数，避免默认线程过度竞争导致卷积异常变慢
torch.set_num_threads(4)
