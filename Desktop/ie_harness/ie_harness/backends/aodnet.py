"""AOD-Net 去雾后端（ICCV 2017, 5 层极轻量 CNN, ~2k 参数）。

权重来源: https://github.com/zhuleyang/aodnet  ``final_model.pth``
模型估计传输图 K，输出 J = K*I - K + b。权重在闭环中冻结。
"""
from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EnhancerBackend


class _AODNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.b = 1.0

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        k = F.relu(self.conv5(cat3))
        output = k * x - k + self.b
        return F.relu(output)


class AODNetBackend(EnhancerBackend):
    name = "aodnet"
    task = "dehaze"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        weight = self.config.get(
            "weight",
            os.path.join(os.path.dirname(__file__), "..", "..", "weights", "aod_final.pth"),
        )
        self.net = _AODNet()
        ckpt = torch.load(weight, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.net.load_state_dict(ckpt)
        self.net.eval()

    def enhance(self, img: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            y = self.net(x)
        y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return self.to_uint8(y * 255.0)
