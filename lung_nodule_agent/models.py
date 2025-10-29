"""Adapters para padronizar a inferência dos modelos de detecção."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from pathlib import Path
import pickle
from collections.abc import Mapping
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
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

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        self.confidence_threshold = confidence_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.display_name = display_name or self.model_path.stem

    def predict(self, image: Image.Image) -> DetectionResult:
        raise NotImplementedError

    @staticmethod
    def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()


class _UltralyticsDetectionAdapter(BaseModelAdapter):
    """Adaptador genérico para modelos carregados via Ultralytics YOLO."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(
            model_path,
            confidence_threshold,
            device,
            display_name=display_name or self.DEFAULT_DISPLAY_NAME,
        )
        if YOLO is None:
            raise ImportError(
                "O pacote 'ultralytics' não está instalado. Instale-o para utilizar este adaptador."
            )
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    DEFAULT_DISPLAY_NAME = "Ultralytics"

    def predict(self, image: Image.Image) -> DetectionResult:  # pragma: no cover - requer modelo real
        results = self.model.predict(
            image,
            conf=self.confidence_threshold,
            verbose=False,
            device=str(self.device),
        )
        if not results:
            return DetectionResult(
                boxes=np.empty((0, 4), dtype=float),
                scores=np.empty((0,), dtype=float),
                labels=np.empty((0,), dtype=int),
                model_name=self.display_name,
            )
        first = results[0]
        boxes = (
            first.boxes.xyxy.cpu().numpy()
            if first.boxes is not None
            else np.empty((0, 4), dtype=float)
        )
        scores = (
            first.boxes.conf.cpu().numpy()
            if first.boxes is not None
            else np.empty((0,), dtype=float)
        )
        labels = (
            first.boxes.cls.cpu().numpy().astype(int)
            if first.boxes is not None
            else np.empty((0,), dtype=int)
        )
        return DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=labels,
            model_name=self.display_name,
        )


class YOLOv8Adapter(_UltralyticsDetectionAdapter):
    """Adaptador para modelos YOLOv8 treinados com Ultralytics."""

    DEFAULT_DISPLAY_NAME = "YOLOv8"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        display_name: Optional[str] = None,
    ):
        super().__init__(
            model_path,
            confidence_threshold,
            device,
            display_name=display_name or self.DEFAULT_DISPLAY_NAME,
        )


def _register_safe_globals() -> bool:
    """Registra classes conhecidas para destravar torch.load em checkpoints customizados."""

    if not hasattr(torch.serialization, "add_safe_globals"):
        return False

    safe_classes = []

    try:  # pragma: no cover - depende da instalação local
        ultralytics_tasks = importlib.import_module("ultralytics.nn.tasks")
        for attr in ("RTDETRDetectionModel", "DetectionModel"):
            if hasattr(ultralytics_tasks, attr):
                safe_classes.append(getattr(ultralytics_tasks, attr))
    except Exception:
        pass

    if not safe_classes:
        return False

    torch.serialization.add_safe_globals(safe_classes)
    return True


def _load_checkpoint(path: Path):
    """Carrega checkpoints lidando com o parâmetro weights_only introduzido no PyTorch 2.6."""

    kwargs = {"map_location": "cpu"}

    signature = inspect.signature(torch.load)
    supports_weights_only = "weights_only" in signature.parameters
    if supports_weights_only:
        kwargs["weights_only"] = True

    def _attempt(**load_kwargs):
        return torch.load(path, **load_kwargs)

    try:
        return _attempt(**kwargs)
    except TypeError as exc:
        if "weights_only" in kwargs:
            kwargs.pop("weights_only")
            return _attempt(**kwargs)
        raise exc
    except (pickle.UnpicklingError, RuntimeError):
        if supports_weights_only and _register_safe_globals():
            try:
                return _attempt(**kwargs)
            except (pickle.UnpicklingError, RuntimeError):
                pass

        if supports_weights_only:
            kwargs["weights_only"] = False
            return _attempt(**kwargs)
        raise


def _looks_like_state_dict(candidate: Any) -> bool:
    """Heurística simples para identificar um state_dict."""

    if isinstance(candidate, Mapping):
        values = list(candidate.values())
        if not values:
            return True
        return all(isinstance(item, torch.Tensor) for item in values)
    return False


def _extract_state_dict(checkpoint: Any):
    """Extrai o state_dict independente do formato salvo no checkpoint."""

    candidates = []

    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            value = checkpoint.get(key)
            if value is not None:
                candidates.append(value)

        for key in ("ema", "model"):
            value = checkpoint.get(key)
            if hasattr(value, "state_dict"):
                candidates.append(value.state_dict())

    if hasattr(checkpoint, "state_dict"):
        candidates.append(checkpoint.state_dict())

    candidates.append(checkpoint)

    for candidate in candidates:
        if _looks_like_state_dict(candidate):
            return candidate

    raise RuntimeError(
        "Não foi possível extrair um state_dict do checkpoint fornecido. "
        "Verifique se o arquivo contém os pesos do modelo no formato esperado."
    )


def _is_ultralytics_state_dict(state_dict: Mapping[str, torch.Tensor], prefixes: Sequence[str]) -> bool:
    if not state_dict:
        return False
    for key in state_dict.keys():
        if any(key.startswith(prefix) for prefix in prefixes):
            return True
    return False


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
        display_name: Optional[str] = None,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        super().__init__(model_path, confidence_threshold, device, display_name=display_name)
        self.model = (
            model_ctor(num_classes=num_classes)
            if num_classes is not None
            else model_ctor()
        )
        if state_dict is None:
            checkpoint = _load_checkpoint(self.model_path)
            state_dict = _extract_state_dict(checkpoint)
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
        return DetectionResult(boxes=boxes, scores=scores, labels=labels, model_name=self.display_name)


def _resolve_detr_model_ctor():
    """Retorna um construtor compatível para o DETR, com fallbacks."""

    errors = []
    try:  # TorchVision >= 0.14
        from torchvision.models.detection import detr_resnet50  # type: ignore[attr-defined]

        return lambda num_classes=None: _instantiate_torchvision_model(
            detr_resnet50, num_classes
        )
    except ImportError as exc:  # pragma: no cover - depende da versão instalada
        errors.append(str(exc))

    try:  # TorchVision 0.13 usa um submódulo dedicado
        from torchvision.models.detection.detr import detr_resnet50  # type: ignore[attr-defined]

        return lambda num_classes=None: _instantiate_torchvision_model(
            detr_resnet50, num_classes
        )
    except ImportError as exc:  # pragma: no cover - depende da versão instalada
        errors.append(str(exc))

    try:  # Fallback via torch.hub (facebookresearch/detr)
        import torch.hub

        def _ctor(num_classes: Optional[int] = None):
            model = torch.hub.load(
                "facebookresearch/detr",
                "detr_resnet50",
                pretrained=False,
            )
            if num_classes is not None and hasattr(model, "class_embed"):
                in_features = model.class_embed.in_features
                model.class_embed = nn.Linear(in_features, num_classes)
            return model

        return _ctor
    except Exception as exc:  # pragma: no cover - depende do ambiente
        errors.append(f"torch.hub fallback failed: {exc}")

    joined = " | ".join(errors)
    raise ImportError(
        "Não foi possível localizar `detr_resnet50` no TorchVision instalado e o fallback via "
        "torch.hub falhou. Atualize para torchvision>=0.13 ou garanta acesso ao repositório "
        "facebookresearch/detr`. Erros encontrados: "
        f"{joined}"
    )


class UltralyticsRTDETRAdapter(_UltralyticsDetectionAdapter):
    """Adaptador para checkpoints RT-DETR treinados com Ultralytics."""

    DEFAULT_DISPLAY_NAME = "DETR"

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        num_classes: int = 2,  # compatibilidade com assinatura do DETRAdapter
        display_name: Optional[str] = None,
    ):
        super().__init__(
            model_path,
            confidence_threshold,
            device,
            display_name=display_name or self.DEFAULT_DISPLAY_NAME,
        )


class _TorchVisionDETRAdapter(_TorchVisionDetectionAdapter):
    """Adaptador para checkpoints DETR compatíveis com TorchVision."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        num_classes: int = 2,
        display_name: Optional[str] = None,
        state_dict: Optional[Mapping[str, torch.Tensor]] = None,
    ):
        model_ctor = _resolve_detr_model_ctor()

        super().__init__(
            model_ctor=model_ctor,
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            num_classes=num_classes,
            display_name=display_name or "DETR",
            state_dict=state_dict,
        )


class DETRAdapter(BaseModelAdapter):  # type: ignore[misc]
    """Escolhe automaticamente o adaptador DETR compatível com o checkpoint fornecido."""

    def __new__(
        cls,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        num_classes: int = 2,
        display_name: Optional[str] = None,
    ):
        display = display_name or "DETR"
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {path}")

        state_dict: Optional[Mapping[str, torch.Tensor]] = None
        try:
            checkpoint = _load_checkpoint(path)
            state_dict = _extract_state_dict(checkpoint)
        except Exception as exc:
            checkpoint_error = exc
        else:
            checkpoint_error = None

        if state_dict is not None and _is_ultralytics_state_dict(state_dict, ("model.",)):
            if YOLO is None:
                raise ImportError(
                    "O checkpoint fornecido parece ser um modelo Ultralytics (RT-DETR), "
                    "mas o pacote 'ultralytics' não está instalado."
                )
            return UltralyticsRTDETRAdapter(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                device=device,
                num_classes=num_classes,
                display_name=display,
            )

        if state_dict is None:
            raise RuntimeError(
                "Falha ao carregar o checkpoint do DETR. "
                f"Erro original: {checkpoint_error}"
            )

        return _TorchVisionDETRAdapter(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device=device,
            num_classes=num_classes,
            display_name=display,
            state_dict=state_dict,
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
        display_name: Optional[str] = None,
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
            display_name=display_name or "Faster R-CNN",
        )