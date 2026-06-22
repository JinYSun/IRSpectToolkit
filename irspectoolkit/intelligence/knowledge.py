"""
知识注入 — 官能团-峰位映射、光谱知识库
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional


# 默认官能团-峰位知识库
DEFAULT_FUNCTIONAL_GROUPS = {
    "O-H (醇)": {"range": (3200, 3550), "shape": "broad", "intensity": "strong",
                 "description": "醇类O-H伸缩振动，通常宽而强"},
    "O-H (羧酸)": {"range": (2500, 3300), "shape": "very_broad", "intensity": "strong",
                   "description": "羧酸O-H伸缩，非常宽，常与C-H重叠"},
    "N-H (胺)": {"range": (3300, 3500), "shape": "medium", "intensity": "medium",
                 "description": "伯胺有双峰，仲胺有单峰"},
    "C-H (烷烃伸缩)": {"range": (2850, 3000), "shape": "sharp", "intensity": "strong",
                      "description": "甲基和亚甲基C-H伸缩振动"},
    "C-H (烯烃伸缩)": {"range": (3000, 3100), "shape": "sharp", "intensity": "weak",
                      "description": "不饱和C-H伸缩，波数高于烷烃"},
    "C-H (芳环伸缩)": {"range": (3000, 3100), "shape": "sharp", "intensity": "weak",
                      "description": "芳环C-H伸缩，常有多个小峰"},
    "C≡N (腈)": {"range": (2200, 2260), "shape": "sharp", "intensity": "medium",
                 "description": "腈基C≡N伸缩，特征性很强"},
    "C≡C (炔)": {"range": (2100, 2260), "shape": "sharp", "intensity": "weak",
                 "description": "炔烃C≡C伸缩"},
    "C=O (酮)": {"range": (1705, 1725), "shape": "sharp", "intensity": "very_strong",
                 "description": "酮类C=O伸缩，IR中最强峰之一"},
    "C=O (醛)": {"range": (1720, 1740), "shape": "sharp", "intensity": "very_strong",
                 "description": "醛类C=O伸缩，常伴有2720/2820的C-H双峰"},
    "C=O (酯)": {"range": (1735, 1750), "shape": "sharp", "intensity": "very_strong",
                 "description": "酯类C=O伸缩"},
    "C=O (酰胺I)": {"range": (1630, 1690), "shape": "sharp", "intensity": "strong",
                   "description": "酰胺C=O伸缩 (酰胺I带)"},
    "C=O (羧酸)": {"range": (1700, 1725), "shape": "sharp", "intensity": "strong",
                   "description": "羧酸C=O伸缩"},
    "C=C (烯)": {"range": (1620, 1680), "shape": "medium", "intensity": "medium",
                 "description": "烯烃C=C伸缩"},
    "C=C (芳环)": {"range": (1450, 1600), "shape": "medium", "intensity": "medium",
                   "description": "芳环骨架振动，通常1500和1600附近有2-3个峰"},
    "N-H (酰胺II)": {"range": (1550, 1640), "shape": "medium", "intensity": "medium",
                     "description": "酰胺N-H弯曲 + C-N伸缩 (酰胺II带)"},
    "CH₂ (弯曲)": {"range": (1450, 1470), "shape": "medium", "intensity": "medium",
                   "description": "亚甲基剪式弯曲"},
    "CH₃ (弯曲)": {"range": (1370, 1390), "shape": "medium", "intensity": "medium",
                   "description": "甲基对称弯曲"},
    "S=O (亚砜)": {"range": (1000, 1070), "shape": "sharp", "intensity": "strong",
                   "description": "亚砜S=O伸缩"},
    "C-O (醇)": {"range": (1000, 1260), "shape": "strong", "intensity": "strong",
                 "description": "醇C-O伸缩"},
    "C-O (酯)": {"range": (1000, 1300), "shape": "strong", "intensity": "strong",
                 "description": "酯C-O-C伸缩，常有双峰"},
    "C-F": {"range": (1000, 1400), "shape": "strong", "intensity": "very_strong",
            "description": "C-F伸缩，非常强"},
    "C-Cl": {"range": (550, 850), "shape": "strong", "intensity": "strong",
             "description": "C-Cl伸缩"},
    "N-O (硝基)": {"range": (1300, 1370), "shape": "strong", "intensity": "very_strong",
                   "description": "硝基对称伸缩"},
    "N-O (硝基不对称)": {"range": (1500, 1570), "shape": "strong", "intensity": "very_strong",
                        "description": "硝基不对称伸缩"},
}


class SpectralKnowledgeBase:
    """
    光谱知识库管理
    
    支持官能团-峰位映射、自定义规则、知识查询
    """

    def __init__(self, knowledge_file: str | None = None):
        if knowledge_file and Path(knowledge_file).exists():
            with open(knowledge_file, "r") as f:
                self.groups = json.load(f)
        else:
            self.groups = DEFAULT_FUNCTIONAL_GROUPS.copy()

        self.rules: list[dict] = []

    def query_by_wavelength(self, wavelength: float, tolerance: float = 10) -> list[dict]:
        """根据波长查询可能的官能团"""
        matches = []
        for name, info in self.groups.items():
            r = info["range"] if isinstance(info, dict) else info
            if r[0] - tolerance <= wavelength <= r[1] + tolerance:
                matches.append({
                    "group": name,
                    "range": r,
                    "description": info.get("description", "") if isinstance(info, dict) else "",
                })
        return matches

    def query_by_keyword(self, keyword: str) -> list[dict]:
        """关键词搜索"""
        keyword_lower = keyword.lower()
        results = []
        for name, info in self.groups.items():
            desc = info.get("description", "") if isinstance(info, dict) else ""
            if keyword_lower in name.lower() or keyword_lower in desc.lower():
                results.append({"group": name, "info": info})
        return results

    def add_group(self, name: str, range_cm: tuple[float, float], **kwargs):
        """添加自定义官能团"""
        self.groups[name] = {"range": range_cm, **kwargs}

    def add_rule(self, condition: str, conclusion: str, confidence: float = 0.8):
        """
        添加分析规则
        
        例: "如果检测到1700cm⁻¹附近有强峰且3300cm⁻¹有宽峰，则可能是羧酸"
        """
        self.rules.append({
            "condition": condition,
            "conclusion": conclusion,
            "confidence": confidence,
        })

    def infer_from_peaks(self, peaks: list[dict]) -> list[dict]:
        """
        根据检测到的峰进行推理
        
        结合峰位、峰形、强度综合判断
        """
        inferences = []
        for peak in peaks:
            wl = peak.get("wavelength", 0)
            height = peak.get("height", 0)
            matches = self.query_by_wavelength(wl)
            for m in matches:
                inferences.append({
                    "wavelength": wl,
                    "group": m["group"],
                    "confidence": 0.7,
                    "reason": f"峰位于 {wl:.0f} cm⁻¹, 在 {m['group']} 的特征范围内",
                })

        # 交叉验证规则
        detected_groups = set(i["group"] for i in inferences)
        for rule in self.rules:
            if any(cond in str(detected_groups) for cond in rule["condition"].split("|")):
                inferences.append({
                    "wavelength": 0,
                    "group": "rule_based",
                    "confidence": rule["confidence"],
                    "reason": rule["conclusion"],
                })

        return inferences

    def save(self, filepath: str):
        """保存知识库到文件"""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.groups, f, ensure_ascii=False, indent=2)

    def __len__(self):
        return len(self.groups)

    def __repr__(self):
        return f"SpectralKnowledgeBase({len(self)} groups, {len(self.rules)} rules)"
