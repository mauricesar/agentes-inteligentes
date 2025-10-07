"""Adapters para padronizar a inferência dos modelos de detecção."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image

try:  # Ultralytics é opcional
    from ultralytics import YOLO
except Exception:  # pragma: no cover - apenas usado quando disponível
    YOLO = None  # type: ignore


@dataclass
class DetectionResult:
    """Representa as saídas normalizadas de um modelo de detecção."""

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    model_name: str

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "boxes": self.boxes,
            "scores": self.scores,
            "labels": self.labels,
            "model_name": np.array([self.model_name] * len(self.boxes)),
        }


class BaseModelAdapter:
    """Classe base para adaptadores de modelos de detecção."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: Optional[str] = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def predict(self, image: Image.Image) -> DetectionResult:
        raise NotImplementedError

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()


class YOLOv8Adapter(BaseModelAdapter):
    """Adaptador para modelos YOLOv8 treinados com Ultralytics."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: Optional[str] = None):
        super().__init__(model_path, confidence_threshold, device)
        if YOLO is None:
            raise ImportError(
                "O pacote 'ultralytics' não está instalado. Instale-o para utilizar o adaptador YOLOv8."
            )
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def predict(self, image: Image.Image) -> DetectionResult:  # pragma: no cover - requer modelo real
        results = self.model.predict(image, conf=self.confidence_threshold, verbose=False, device=str(self.device))
        if not results:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=float),
                scores=np.empty((0,), dtype=float),
                labels=np.empty((0,), dtype=int),
                model_name=self.model_path.stem,
            )
        first = results[0]
        boxes = first.boxes.xyxy.cpu().numpy() if first.boxes is not None else np.empty((0, 4), dtype=float)
        scores = first.boxes.conf.cpu().numpy() if first.boxes is not None else np.empty((0,), dtype=float)
        labels = first.boxes.cls.cpu().numpy().astype(int) if first.boxes is not None else np.empty((0,), dtype=int)
        return DetectionResult(boxes=boxes, scores=scores, labels=labels, model_name=self.model_path.stem)


def _instantiate_torchvision_model(model_fn, num_classes: Optional[int]):
    """Cria o modelo tratando diferenças de assinatura entre versões do TorchVision."""

    kwargs = {}
    signature = inspect.signature(model_fn)
    parameters = signature.parameters

    if num_classes is not None and "num_classes" in parameters:
        kwargs["num_classes"] = num_classes
    if "weights" in parameters:
        kwargs["weights"] = None
    elif "pretrained" in parameters:
        kwargs["pretrained"] = False

    return model_fn(**kwargs)


class _TorchVisionDetectionAdapter(BaseModelAdapter):
    """Classe base para modelos TorchVision."""

    def __init__(
        self,
        model_ctor,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        super().__init__(model_path, confidence_threshold, device)
        self.model = (
            model_ctor(num_classes=num_classes)
            if num_classes is not None
            else model_ctor()
        )
        checkpoint = torch.load(self.model_path, map_location="cpu")
        if "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self._default_transform

    @staticmethod
    def _default_transform(image: Image.Image) -> torch.Tensor:
        array = np.asarray(image.convert("RGB")) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).float()
        return tensor

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> DetectionResult:
        tensor = self.transform(image).to(self.device)
        output = self.model([tensor])[0]
        scores = output["scores"].detach().cpu().numpy()
        keep = scores >= self.confidence_threshold
        boxes = output["boxes"].detach().cpu().numpy()[keep]
        scores = scores[keep]
        labels = output["labels"].detach().cpu().numpy()[keep].astype(int)
        return DetectionResult(boxes=boxes, scores=scores, labels=labels, model_name=self.model_path.stem)


def _import_detr_resnet50():
    """Tenta resolver a função ``detr_resnet50`` em diferentes versões do TorchVision."""

    errors = []
    try:  # TorchVision >= 0.14
        from torchvision.models.detection import detr_resnet50  # type: ignore[attr-defined]

        return detr_resnet50
    except ImportError as exc:  # pragma: no cover - depende da versão instalada
        errors.append(str(exc))

    try:  # TorchVision 0.13 usa um submódulo dedicado
        from torchvision.models.detection.detr import detr_resnet50  # type: ignore[attr-defined]

        return detr_resnet50
    except ImportError as exc:  # pragma: no cover - depende da versão instalada
        errors.append(str(exc))

    joined = " | ".join(errors)
    raise ImportError(
        "Não foi possível localizar `detr_resnet50` no TorchVision instalado. "
        "Atualize para torchvision>=0.13 ou forneça manualmente um construtor DETR personalizado. "
        "Erros encontrados: "
        f"{joined}"
    )


class DETRAdapter(_TorchVisionDetectionAdapter):
    """Adaptador para modelos DETR fine-tunados."""

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, device: Optional[str] = None, num_classes: int = 2):
        detr_resnet50 = _import_detr_resnet50()

        super().__init__(
            model_ctor=lambda num_classes=num_classes: _instantiate_torchvision_model(
                detr_resnet50, num_classes
            ),
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            num_classes=num_classes,
        )


class FasterRCNNAdapter(_TorchVisionDetectionAdapter):
    """Adaptador para modelos Faster R-CNN."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        num_classes: int = 2,
        backbone: str = "resnet50",
    ):
        from torchvision.models.detection import fasterrcnn_resnet50_fpn

        if backbone != "resnet50":  # pragma: no cover - placeholder para extensões futuras
            raise ValueError("Atualmente apenas o backbone resnet50 é suportado para o adaptador padrão.")
        super().__init__(
            model_ctor=lambda num_classes=num_classes: _instantiate_torchvision_model(
                fasterrcnn_resnet50_fpn, num_classes
            ),
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            num_classes=num_classes,
        )
