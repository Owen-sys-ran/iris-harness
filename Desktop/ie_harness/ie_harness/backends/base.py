"""统一增强后端接口。

Harness 只负责外围流水线（预处理 / 后处理 / 评测 / 调度），
后端（policy model）权重在整个 CRSI 闭环中冻结，不做任何更新。
"""
from __future__ import annotations

import abc
import numpy as np


class EnhancerBackend(abc.ABC):
    """所有增强后端的抽象基类。

    输入 / 输出均为 ``uint8``、``HWC``、RGB 顺序的 numpy 数组。
    """

    #: 后端唯一标识，注册到 registry 时使用
    name: str = "base"
    #: 主攻退化类型: lowlight / underwater / dehaze / denoise / wb / contrast
    task: str = "generic"

    def __init__(self, config: dict | None = None):
        self.config = config or {}

    @abc.abstractmethod
    def enhance(self, img: np.ndarray) -> np.ndarray:
        """对单张 uint8 RGB 图像做增强，返回 uint8 RGB 图像。"""

    # ---- 通用工具 ----
    @staticmethod
    def to_uint8(x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x), 0, 255).astype(np.uint8)
