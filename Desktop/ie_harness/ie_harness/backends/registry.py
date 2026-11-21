"""后端注册表：按名称实例化，Harness 配置只引用名称与参数。"""
from __future__ import annotations

import json

from .base import EnhancerBackend
from .zero_dce import ZeroDCEBackend
from .funie_gan import FunIEGANBackend
from .aodnet import AODNetBackend
from .classical import (
    GrayWorldWBBackend,
    CLAHEBackend,
    MSRBackend,
    BilateralDenoiseBackend,
    NLMDenoiseBackend,
)

_REGISTRY: dict[str, type[EnhancerBackend]] = {
    ZeroDCEBackend.name: ZeroDCEBackend,
    FunIEGANBackend.name: FunIEGANBackend,
    AODNetBackend.name: AODNetBackend,
    GrayWorldWBBackend.name: GrayWorldWBBackend,
    CLAHEBackend.name: CLAHEBackend,
    MSRBackend.name: MSRBackend,
    BilateralDenoiseBackend.name: BilateralDenoiseBackend,
    NLMDenoiseBackend.name: NLMDenoiseBackend,
}

# 权重冻结、后端无状态，按 (名称, 配置) 缓存实例，避免重复加载权重
_CACHE: dict[tuple, EnhancerBackend] = {}

# 任务 -> 推荐后端（供 Proposer 按退化类型选择）
TASK_BACKENDS: dict[str, list[str]] = {
    "lowlight": ["zero_dce", "msr", "clahe"],
    "underwater": ["funie_gan", "grayworld", "msr"],
    "dehaze": ["aodnet", "msr", "clahe"],
    "denoise": ["nlm", "bilateral"],
}


def available_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


def build_backend(name: str, config: dict | None = None) -> EnhancerBackend:
    if name not in _REGISTRY:
        raise KeyError(f"未知后端: {name}; 可选: {available_backends()}")
    key = (name, json.dumps(config or {}, sort_keys=True))
    if key not in _CACHE:
        _CACHE[key] = _REGISTRY[name](config)
    return _CACHE[key]
