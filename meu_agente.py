"""Script de exemplo para executar o agente decisor com seus modelos treinados.

Edite os caminhos abaixo antes de rodar:

- `YOLOV8_WEIGHTS`, `DETR_WEIGHTS` e `FASTER_RCNN_WEIGHTS`: apontam para os arquivos de pesos
  treinados por você.
- `IMAGE_PATH` ou `DATASET_DIR`: defina apenas um deles para escolher entre rodar uma única
  imagem ou um diretório com várias imagens.
- `FEEDBACK_LOG`: arquivo onde os resultados serão registrados automaticamente para consulta
  posterior ou versionamento.

Após atualizar os caminhos, execute:

```bash
python meu_agente.py
```
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from lung_nodule_agent import (
    DETRAdapter,
    FasterRCNNAdapter,
    LungNoduleDecisionAgent,
    YOLOv8Adapter,
    export_feedback,
    save_visualizations,
)

# ---------------------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------------------
YOLOV8_WEIGHTS = Path("C:/Users/Mauricésar/Desktop/extracao/yolov8/best_model.pt")
#DETR_WEIGHTS = Path("/caminho/para/seu_modelo_detr.pth")
FASTER_RCNN_WEIGHTS = Path("C:/Users/Mauricésar/Desktop/extracao/fasterrcnn/fasterrcnn/fasterrcnn_resnet50_fpn_best.pth")

# Defina o caminho para uma imagem individual OU para um diretório de imagens.
IMAGE_PATH: Optional[Path] = Path("C:/Users/Mauricésar/Desktop/extracao/validate/images/2a2d24f9e9397ea669cf6ec9f850c367.png")
DATASET_DIR: Optional[Path] = None  # Exemplo: Path("/caminho/para/pasta_de_testes")

# Arquivo onde os feedbacks serão registrados automaticamente.
FEEDBACK_LOG = Path("feedbacks2.jsonl")

# Diretório base para salvar visualizações com as caixas desenhadas.
VISUALIZATIONS_DIR: Optional[Path] = Path("C:/Users\Mauricésar/Desktop/teste/agentes-inteligentes/visualization")

# Ajuste se quiser alterar os limiares usados pelo agente.
CONFIDENCE_THRESHOLD = 0.25
VOTE_THRESHOLD: Optional[int] = None
IOU_THRESHOLD = 0.5

# ---------------------------------------------------------------------------


def _validate_paths(paths: Iterable[Optional[Path]]) -> None:
    missing: List[Path] = [path for path in paths if path is not None and not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(
            "Os caminhos abaixo não foram encontrados. Atualize o arquivo antes de executar:\n"
            f"{formatted}"
        )


def main() -> None:
    if IMAGE_PATH and DATASET_DIR:
        raise SystemExit(
            "Configure apenas IMAGE_PATH ou DATASET_DIR. Ambos definidos ao mesmo tempo não são suportados."
        )

    _validate_paths([YOLOV8_WEIGHTS, FASTER_RCNN_WEIGHTS]) #_validate_paths([YOLOV8_WEIGHTS, DETR_WEIGHTS, FASTER_RCNN_WEIGHTS])

    adapters = [
        YOLOv8Adapter(YOLOV8_WEIGHTS),
        #DETRAdapter(DETR_WEIGHTS),
        FasterRCNNAdapter(FASTER_RCNN_WEIGHTS),
    ]

    agent = LungNoduleDecisionAgent(
        adapters,
        iou_threshold=IOU_THRESHOLD,
        vote_threshold=VOTE_THRESHOLD,
        min_score=CONFIDENCE_THRESHOLD,
    )

    decisions = []
    metadata = {
        "models": {
            "yolov8": str(YOLOV8_WEIGHTS),
            #"detr": str(DETR_WEIGHTS),
            "faster_rcnn": str(FASTER_RCNN_WEIGHTS),
        },
        "iou_threshold": IOU_THRESHOLD,
        "vote_threshold": VOTE_THRESHOLD,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }

    if DATASET_DIR:
        _validate_paths([DATASET_DIR])
        decisions = agent.evaluate_directory(str(DATASET_DIR))
        metadata["dataset_dir"] = str(DATASET_DIR)
    elif IMAGE_PATH:
        _validate_paths([IMAGE_PATH])
        decision = agent.predict(str(IMAGE_PATH))
        decisions = [decision]
        metadata["image_path"] = str(IMAGE_PATH)
    else:
        raise SystemExit("Defina IMAGE_PATH ou DATASET_DIR antes de executar este script.")

    if VISUALIZATIONS_DIR is not None:
        metadata["visualizations_dir"] = str(VISUALIZATIONS_DIR)

    print(json.dumps([decision.to_dict() for decision in decisions], indent=2, ensure_ascii=False))

    export_feedback(decisions, FEEDBACK_LOG, metadata=metadata, append=True)
    print(f"Feedback registrado em: {FEEDBACK_LOG.resolve()}")

    if VISUALIZATIONS_DIR is not None:
        save_visualizations(decisions, VISUALIZATIONS_DIR)
        print(
            "Imagens anotadas salvas em: "
            f"{VISUALIZATIONS_DIR.resolve()}"
        )


if __name__ == "__main__":
    main()