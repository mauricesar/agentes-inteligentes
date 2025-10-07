"""Pacote para integração de modelos de detecção de nódulos pulmonares."""

from .decision_agent import LungNoduleDecisionAgent, AgentDecision
from .models import (
    BaseModelAdapter,
    YOLOv8Adapter,
    DETRAdapter,
    FasterRCNNAdapter,
    DetectionResult,
)

__all__ = [
    "LungNoduleDecisionAgent",
    "AgentDecision",
    "BaseModelAdapter",
    "YOLOv8Adapter",
    "DETRAdapter",
    "FasterRCNNAdapter",
    "DetectionResult",
]
