"""Interface de linha de comando para o agente decisor de nódulos pulmonares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from lung_nodule_agent import (
    BaseModelAdapter,
    DETRAdapter,
    FasterRCNNAdapter,
    LungNoduleDecisionAgent,
    YOLOv8Adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa o agente decisor em uma imagem ou diretório.")
    parser.add_argument("--yolo", type=str, help="Caminho para o modelo YOLOv8 treinado." )
    parser.add_argument("--detr", type=str, help="Caminho para o checkpoint DETR.")
    parser.add_argument("--faster-rcnn", type=str, help="Caminho para o checkpoint Faster R-CNN.")
    parser.add_argument("--image", type=str, help="Imagem única para inferência.")
    parser.add_argument(
        "--dataset",
        type=str,
        help="Diretório contendo imagens de raio-x para processamento em lote.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Arquivo JSON opcional para salvar o resultado.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.25,
        help="Limiar mínimo de confiança para considerar uma detecção individual.",
    )
    parser.add_argument(
        "--vote-threshold",
        type=int,
        default=None,
        help="Quantidade mínima de modelos que devem concordar para um veredito positivo.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU mínimo para que detecções de modelos diferentes sejam fundidas.",
    )
    return parser.parse_args()


def build_agent(args: argparse.Namespace) -> LungNoduleDecisionAgent:
    adapters: List[BaseModelAdapter] = []
    if args.yolo:
        adapters.append(YOLOv8Adapter(args.yolo, confidence_threshold=args.confidence_threshold))
    if args.detr:
        adapters.append(DETRAdapter(args.detr, confidence_threshold=args.confidence_threshold))
    if args.faster_rcnn:
        adapters.append(FasterRCNNAdapter(args.faster_rcnn, confidence_threshold=args.confidence_threshold))
    if not adapters:
        raise SystemExit("Nenhum modelo informado. Utilize --yolo, --detr ou --faster-rcnn.")
    return LungNoduleDecisionAgent(
        adapters,
        iou_threshold=args.iou_threshold,
        vote_threshold=args.vote_threshold,
        min_score=args.confidence_threshold,
    )


def main() -> None:
    args = parse_args()
    agent = build_agent(args)
    outputs = []
    if args.image:
        decision = agent.predict(args.image)
        outputs.append(decision.to_dict())
    if args.dataset:
        decisions = agent.evaluate_directory(args.dataset)
        outputs.extend(decision.to_dict() for decision in decisions)
    if not outputs:
        raise SystemExit("Informe --image ou --dataset para executar a inferência.")
    if args.output:
        Path(args.output).write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
