"""Pacote para integração de modelos de detecção de nódulos pulmonares."""

from .decision_agent import LungNoduleDecisionAgent, AgentDecision, export_feedback, save_visualizations
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
    "export_feedback",
    "save_visualizations",
    "BaseModelAdapter",
    "YOLOv8Adapter",
    "DETRAdapter",
    "FasterRCNNAdapter",
    "DetectionResult",
]