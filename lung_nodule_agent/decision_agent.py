"""Agente decisor para combinar modelos de detecção de nódulos pulmonares."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from .models import BaseModelAdapter, DetectionResult


@dataclass
class AggregatedDetection:
    """Detecção resultante da fusão entre os modelos."""

    box: np.ndarray
    score: float
    label: int
    supporting_models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "box": self.box.tolist(),
            "score": float(self.score),
            "label": int(self.label),
            "supporting_models": self.supporting_models,
        }


@dataclass
class AgentDecision:
    """Resultado final do agente."""

    image_path: str
    model_results: Dict[str, DetectionResult]
    aggregated_detections: List[AggregatedDetection]
    verdict: str
    confidence: float
    explanation: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "image_path": self.image_path,
            "verdict": self.verdict,
            "confidence": float(self.confidence),
            "explanation": self.explanation,
            "aggregated_detections": [det.to_dict() for det in self.aggregated_detections],
            "model_results": {
                name: {
                    "boxes": result.boxes.tolist(),
                    "scores": result.scores.tolist(),
                    "labels": result.labels.tolist(),
                }
                for name, result in self.model_results.items()
            },
        }


class LungNoduleDecisionAgent:
    """Combina modelos de detecção para gerar uma decisão única."""

    def __init__(
        self,
        model_adapters: Sequence[BaseModelAdapter],
        iou_threshold: float = 0.5,
        vote_threshold: Optional[int] = None,
        min_score: float = 0.25,
    ) -> None:
        if not model_adapters:
            raise ValueError("Ao menos um adaptador de modelo deve ser informado.")
        self.model_adapters = list(model_adapters)
        self.iou_threshold = iou_threshold
        self.vote_threshold = vote_threshold or max(1, len(self.model_adapters) // 2 + 1)
        self.min_score = min_score

    def predict(self, image_path: str) -> AgentDecision:
        image = Image.open(image_path).convert("RGB")
        model_results: Dict[str, DetectionResult] = {}
        for adapter in self.model_adapters:
            result = adapter.predict(image)
            model_results[adapter.model_path.stem] = result
        aggregated = self._aggregate(model_results)
        verdict, confidence, explanation = self._decide(aggregated)
        return AgentDecision(
            image_path=str(image_path),
            model_results=model_results,
            aggregated_detections=aggregated,
            verdict=verdict,
            confidence=confidence,
            explanation=explanation,
        )

    def evaluate_directory(self, image_dir: str, extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".bmp")) -> List[AgentDecision]:
        path = Path(image_dir)
        if not path.exists():
            raise FileNotFoundError(f"Diretório {image_dir} não encontrado")
        decisions: List[AgentDecision] = []
        for image_path in sorted(path.iterdir()):
            if image_path.suffix.lower() not in extensions:
                continue
            decisions.append(self.predict(str(image_path)))
        return decisions

    def _aggregate(self, model_results: Dict[str, DetectionResult]) -> List[AggregatedDetection]:
        clusters: List[Dict[str, object]] = []
        for model_name, result in model_results.items():
            for box, score, label in zip(result.boxes, result.scores, result.labels):
                if score < self.min_score:
                    continue
                cluster = self._find_cluster(clusters, box, label)
                if cluster is None:
                    cluster = {
                        "boxes": [box],
                        "scores": [float(score)],
                        "labels": [int(label)],
                        "models": {model_name},
                    }
                    clusters.append(cluster)
                else:
                    cluster["boxes"].append(box)  # type: ignore[index]
                    cluster["scores"].append(float(score))  # type: ignore[index]
                    cluster["labels"].append(int(label))  # type: ignore[index]
                    cluster["models"].add(model_name)  # type: ignore[index]
        aggregated: List[AggregatedDetection] = []
        for cluster in clusters:
            boxes = np.stack(cluster["boxes"])  # type: ignore[index]
            scores = np.array(cluster["scores"], dtype=float)  # type: ignore[index]
            labels = cluster["labels"]  # type: ignore[index]
            models = sorted(cluster["models"])  # type: ignore[index]
            weights = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
            fused_box = np.average(boxes, axis=0, weights=weights)
            fused_score = float(np.average(scores, weights=weights))
            fused_label = max(set(labels), key=labels.count)
            aggregated.append(
                AggregatedDetection(
                    box=fused_box,
                    score=fused_score,
                    label=fused_label,
                    supporting_models=models,
                )
            )
        aggregated.sort(key=lambda det: det.score, reverse=True)
        return aggregated

    def _find_cluster(
        self,
        clusters: List[Dict[str, object]],
        candidate_box: np.ndarray,
        candidate_label: int,
    ) -> Optional[Dict[str, object]]:
        for cluster in clusters:
            boxes = cluster["boxes"]  # type: ignore[index]
            labels = cluster["labels"]  # type: ignore[index]
            if labels and labels[0] != candidate_label:
                continue
            for existing_box in boxes:
                if self._iou(existing_box, candidate_box) >= self.iou_threshold:
                    return cluster
        return None

    @staticmethod
    def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        xa1, ya1, xa2, ya2 = box_a
        xb1, yb1, xb2, yb2 = box_b
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
        union = area_a + area_b - inter_area
        return inter_area / union if union > 0 else 0.0

    def _decide(self, detections: List[AggregatedDetection]) -> tuple[str, float, str]:
        if not detections:
            return (
                "Sem evidências de nódulos",
                0.0,
                "Nenhum dos modelos detectou regiões suspeitas acima do limiar configurado.",
            )
        supporting_votes = [len(det.supporting_models) for det in detections]
        best_detection = detections[0]
        if max(supporting_votes) >= self.vote_threshold:
            confidence = min(1.0, best_detection.score)
            explanation = (
                "Detecção consistente obtida pelos modelos: "
                + ", ".join(best_detection.supporting_models)
            )
            return "Provável presença de nódulo", confidence, explanation
        confidence = mean([det.score for det in detections]) if detections else 0.0
        explanation = (
            "Foram encontradas detecções, porém sem consenso suficiente. "
            "Considere revisão manual."
        )
        return "Inconclusivo", float(confidence), explanation
