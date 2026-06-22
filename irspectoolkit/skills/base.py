"""Skill基类和统一返回格式"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SkillResult:
    """Skill统一返回格式"""
    conclusion: str = ""           # 一句话结论
    confidence: float = 0.0        # 置信度 0-1
    details: dict = field(default_factory=dict)    # 详细数据
    figures: list = field(default_factory=list)    # 生成的图表
    report: str = ""               # 可读报告
    raw_data: dict = field(default_factory=dict)   # 原始计算结果

    def __repr__(self):
        return f"SkillResult(conclusion='{self.conclusion}', confidence={self.confidence:.2f})"


class SpectralSkill:
    """Skill基类"""
    name: str = "base"
    description: str = "Base skill"

    def run(self, spectra, **kwargs) -> SkillResult:
        raise NotImplementedError
