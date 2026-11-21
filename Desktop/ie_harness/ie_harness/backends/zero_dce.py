"""Zero-DCE 低光增强后端（ICCV 2020, 7 层轻量 CNN, ~0.08M 参数）。

权重来源: https://github.com/Li-Chongyi/Zero-DCE  ``Epoch99.pth``
权重在闭环中冻结；本类只负责加载与前向推理。

性能备注: 该 checkpoint 的权重值会触发 oneDNN(MKL-DNN) 的低效卷积内核，
CPU 上全分辨率推理异常变慢；因此推理时局部关闭 mkldnn，并内部缩放到
max_side 分辨率（贴近训练分辨率），输出再放回原尺寸。
"""
from __future__ import annotations

import contextlib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import EnhancerBackend


@contextlib.contextmanager
def _no_mkldnn():
    old = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = old


class _EnhanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        relu = nn.ReLU(inplace=True)
        n = 32
        self.e_conv1 = nn.Conv2d(3, n, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(n, n, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(n * 2, n, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(n * 2, n, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(n * 2, 24, 3, 1, 1, bias=True)
        self.relu = relu

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r = torch.split(x_r, 3, dim=1)

        x = x + r[0] * (torch.pow(x, 2) - x)
        x = x + r[1] * (torch.pow(x, 2) - x)
        x = x + r[2] * (torch.pow(x, 2) - x)
        e1 = x + r[3] * (torch.pow(x, 2) - x)
        x = e1 + r[4] * (torch.pow(e1, 2) - e1)
        x = x + r[5] * (torch.pow(x, 2) - x)
        x = x + r[6] * (torch.pow(x, 2) - x)
        e2 = x + r[7] * (torch.pow(x, 2) - x)
        return e1, e2


class ZeroDCEBackend(EnhancerBackend):
    name = "zero_dce"
    task = "lowlight"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        weight = self.config.get(
            "weight",
            os.path.join(os.path.dirname(__file__), "..", "..", "weights", "zero_dce_epoch99.pth"),
        )
        self.net = _EnhanceNet()
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.net.load_state_dict(ckpt)
        self.net.eval()
        self.max_side = int(self.config.get("max_side", 256))

    def enhance(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        scale = self.max_side / max(h, w)
        if scale < 1.0:
            nw, nh = int(w * scale), int(h * scale)
            inp = np.asarray(Image.fromarray(img).resize((nw, nh), Image.BICUBIC))
        else:
            inp = img
        x = torch.from_numpy(inp.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with _no_mkldnn(), torch.no_grad():
            _, out = self.net(x)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = self.to_uint8(out * 255.0)
        if (h, w) != out.shape[:2]:
            out = np.asarray(Image.fromarray(out).resize((w, h), Image.BICUBIC))
        return out
