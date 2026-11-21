"""FUnIE-GAN 水下图像增强后端（RA-L 2020, 5 层 UNet 条件 GAN 生成器）。

权重来源: https://github.com/xahidbuffon/funie-gan  ``funie_generator.pth``
训练分辨率 256x256；推理时 resize 到 256，输出再映射回原尺寸。
权重在闭环中冻结。
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .base import EnhancerBackend


class _UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class _UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat((x, skip), 1)


class _Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.down1 = _UNetDown(in_channels, 32, bn=False)
        self.down2 = _UNetDown(32, 128)
        self.down3 = _UNetDown(128, 256)
        self.down4 = _UNetDown(256, 256)
        self.down5 = _UNetDown(256, 256, bn=False)
        self.up1 = _UNetUp(256, 256)
        self.up2 = _UNetUp(512, 256)
        self.up3 = _UNetUp(512, 128)
        self.up4 = _UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final(u4)


class FunIEGANBackend(EnhancerBackend):
    name = "funie_gan"
    task = "underwater"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        weight = self.config.get(
            "weight",
            os.path.join(os.path.dirname(__file__), "..", "..", "weights", "funie_generator.pth"),
        )
        self.net = _Generator()
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.net.load_state_dict(ckpt)
        self.net.eval()
        self.size = int(self.config.get("size", 256))

    def enhance(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        pil = Image.fromarray(img).resize((self.size, self.size), Image.BICUBIC)
        x = np.asarray(pil).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5  # Normalize((0.5,)*3, (0.5,)*3)
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            y = self.net(t)
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        y = (y * 0.5 + 0.5) * 255.0
        out = self.to_uint8(y)
        if (h, w) != (self.size, self.size):
            out = np.asarray(Image.fromarray(out).resize((w, h), Image.BICUBIC))
        return out
