"""知识驱动分析Skills"""
from __future__ import annotations
import numpy as np
from .base import SkillResult
from irspectoolkit.analysis.peak import detect_peaks, assign_functional_groups
from irspectoolkit.intelligence.knowledge import SpectralKnowledgeBase


def functional_group_scan(
    spectrum: np.ndarray,
    wavelengths: np.ndarray,
    knowledge_base: SpectralKnowledgeBase | None = None,
) -> SkillResult:
    """
    官能团扫描: 光谱里有哪些官能团？
    """
    if knowledge_base is None:
        knowledge_base = SpectralKnowledgeBase()

    # 峰检测
    peaks = detect_peaks(spectrum, wavelengths)

    # 峰归属
    assignments = assign_functional_groups(peaks, knowledge_base.groups)

    # 知识推理
    inferences = knowledge_base.infer_from_peaks(peaks)

    # 汇总
    detected_groups = set()
    for a in assignments:
        detected_groups.update(a.get("assignments", []))
    detected_groups.discard("unassigned")

    conclusion = f"检测到 {len(peaks)} 个峰, 识别出 {len(detected_groups)} 种官能团"
    if detected_groups:
        conclusion += f": {', '.join(sorted(detected_groups))}"

    report_lines = [conclusion, "\n峰归属详情:"]
    for a in assignments:
        groups_str = ", ".join(a["assignments"])
        report_lines.append(f"  {a['wavelength']:.0f} cm⁻¹ → {groups_str}")

    if inferences:
        report_lines.append("\n知识推理:")
        for inf in inferences[:5]:
            report_lines.append(f"  {inf['reason']} (置信度: {inf['confidence']:.0%})")

    return SkillResult(
        conclusion=conclusion,
        confidence=len(detected_groups) / max(len(peaks), 1),
        details={
            "peaks": peaks,
            "assignments": assignments,
            "detected_groups": sorted(detected_groups),
            "inferences": inferences,
        },
        report="\n".join(report_lines),
        raw_data={"peaks": peaks, "assignments": assignments},
    )
