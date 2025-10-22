"""Agente decisor para combinar modelos de detecção de nódulos pulmonares."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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


def export_feedback(
    decisions: Sequence[AgentDecision],
    destination: str | Path,
    metadata: Optional[Mapping[str, Any]] = None,
    append: bool = True,
) -> None:
    """Registra as decisões do agente em um arquivo JSON/JSONL.

    Args:
        decisions: Coleção de decisões produzidas pelo agente.
        destination: Caminho do arquivo que receberá o log.
        metadata: Informações adicionais sobre a execução (ex.: caminhos
            dos modelos, parâmetros ou notas do operador).
        append: Quando ``True`` (padrão) adiciona a entrada ao final do arquivo
            em formato JSONL; quando ``False`` sobrescreve o arquivo com um
            único objeto JSON contendo a execução.
    """

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "decisions": [decision.to_dict() for decision in decisions],
    }

    path = Path(destination)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    if append:
        mode = "a" if path.exists() else "w"
        with path.open(mode, encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False))
            file.write("\n")
    else:
        with path.open("w", encoding="utf-8") as file:
            json.dump(entry, file, indent=2, ensure_ascii=False)


def save_visualizations(
    decisions: Sequence[AgentDecision],
    destination: str | Path,
    fusion_subdir: str = "fusion",
    line_width: int = 3,
) -> None:
    """Gera imagens anotadas com as detecções individuais e agregadas.

    Para cada decisão, cria uma cópia anotada da imagem original em um diretório com o
    nome do modelo que originou as caixas. Uma imagem adicional contendo as detecções
    fundidas é salva dentro do subdiretório definido em ``fusion_subdir``.

    Args:
        decisions: Coleção de decisões retornadas pelo agente.
        destination: Diretório raiz onde as imagens anotadas serão criadas.
        fusion_subdir: Nome da pasta que armazenará as anotações da fusão.
        line_width: Espessura das caixas desenhadas nas imagens resultantes.
    """

    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    for decision in decisions:
        image_path = Path(decision.image_path)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Imagem original {decision.image_path} não encontrada para gerar visualizações."
            )

        with Image.open(image_path) as original:
            base_image = original.convert("RGB")

        image_name = image_path.name

        for model_name, result in decision.model_results.items():
            model_dir = root / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            annotated = _draw_detections(
                base_image.copy(),
                result.boxes,
                result.scores,
                result.labels,
                color=(255, 0, 0),
                label_builder=lambda score, label: f"score={score:.2f} label={label}",
                line_width=line_width,
            )
            annotated.save(model_dir / image_name)

        fusion_dir = root / fusion_subdir
        fusion_dir.mkdir(parents=True, exist_ok=True)
        fused_image = base_image.copy()
        fused_image = _draw_aggregated(
            fused_image,
            decision.aggregated_detections,
            line_width=line_width,
        )
        fused_image.save(fusion_dir / image_name)


def _draw_detections(
    image: Image.Image,
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    color: tuple[int, int, int],
    label_builder: Callable[[float, int], str],
    line_width: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _load_font()

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(float, box)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=line_width)
        caption = label_builder(float(score), int(label))
        if caption and font is not None:
            _draw_label(
                draw,
                font,
                caption,
                x1,
                y1,
                background_color=color,
                text_color=(255, 255, 255),
            )

    return image


def _draw_aggregated(
    image: Image.Image,
    detections: Sequence[AggregatedDetection],
    *,
    line_width: int,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = _load_font()

    for det in detections:
        x1, y1, x2, y2 = map(float, det.box)
        draw.rectangle([(x1, y1), (x2, y2)], outline=(46, 204, 113), width=line_width)
        if font is not None:
            models = ", ".join(det.supporting_models)
            caption = f"score={det.score:.2f}"
            if models:
                caption += f"\nmodelos: {models}"
            _draw_label(
                draw,
                font,
                caption,
                x1,
                y1,
                background_color=(46, 204, 113),
                text_color=(0, 0, 0),
            )

    return image


def _load_font() -> Optional[ImageFont.ImageFont]:
    try:
        return ImageFont.load_default()
    except Exception:  # pragma: no cover - fallback defensivo
        return None


def _draw_label(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    caption: str,
    x1: float,
    y1: float,
    *,
    background_color: tuple[int, int, int],
    text_color: tuple[int, int, int],
) -> None:
    if not caption:
        return

    lines = caption.split("\n")
    line_sizes = [_measure_text(font, line) for line in lines]
    text_width = max((width for width, _ in line_sizes), default=0)
    text_height = sum((height for _, height in line_sizes))
    line_spacing = 2
    padding = 2

    if len(lines) > 1:
        text_height += line_spacing * (len(lines) - 1)

    text_x = x1
    text_y = max(0.0, y1 - text_height - padding * 2)

    background = [
        text_x,
        text_y,
        text_x + text_width + padding * 2,
        text_y + text_height + padding * 2,
    ]
    draw.rectangle(background, fill=background_color)

    current_y = text_y + padding
    for index, (line, (_, height)) in enumerate(zip(lines, line_sizes)):
        draw.text((text_x + padding, current_y), line, fill=text_color, font=font)
        current_y += height
        if index < len(lines) - 1:
            current_y += line_spacing


def _measure_text(font: ImageFont.ImageFont, text: str) -> tuple[int, int]:
    """Retorna a largura e altura de ``text`` utilizando a ``font`` fornecida."""

    try:
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
    except AttributeError:
        # Compatibilidade com versões mais antigas do Pillow que ainda expõem ``getsize``.
        width, height = font.getsize(text)
        return int(width), int(height)