"""Interface de linha de comando para o agente decisor de nódulos pulmonares."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from lung_nodule_agent import (
    BaseModelAdapter,
    DETRAdapter,
    FasterRCNNAdapter,
    LungNoduleDecisionAgent,
    YOLOv8Adapter,
    export_feedback,
    save_visualizations,
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
        "--log-feedback",
        type=str,
        help=(
            "Arquivo JSONL para registrar a execução (útil para versionar ou "
            "compartilhar histórico de testes)."
        ),
    )
    parser.add_argument(
        "--feedback-note",
        type=str,
        help="Observação opcional a ser registrada junto ao feedback (ex.: condições do teste).",
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
    parser.add_argument(
        "--visualizations-dir",
        type=str,
        help=(
            "Diretório onde serão salvas imagens anotadas por modelo e a fusão final. "
            "Serão criados subdiretórios com o nome de cada modelo e um subdiretório adicional "
            "para a fusão."
        ),
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
    decisions = []
    if args.image:
        decision = agent.predict(args.image)
        decisions.append(decision)
    if args.dataset:
        decisions.extend(agent.evaluate_directory(args.dataset))
    if not decisions:
        raise SystemExit("Informe --image ou --dataset para executar a inferência.")
    outputs = [decision.to_dict() for decision in decisions]
    if args.output:
        Path(args.output).write_text(json.dumps(outputs, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
    if args.visualizations_dir:
        save_visualizations(decisions, args.visualizations_dir)
        print(
            "Visualizações salvas em: "
            f"{Path(args.visualizations_dir).resolve()}"
        )
    if args.log_feedback:
        metadata: Dict[str, object] = {
            "models": {
                key: value
                for key, value in {
                    "yolo": args.yolo,
                    "detr": args.detr,
                    "faster_rcnn": args.faster_rcnn,
                }.items()
                if value
            },
            "inputs": {
                "image": args.image,
                "dataset": args.dataset,
            },
            "parameters": {
                "confidence_threshold": args.confidence_threshold,
                "vote_threshold": args.vote_threshold,
                "iou_threshold": args.iou_threshold,
            },
        }
        if args.output:
            metadata["output_file"] = args.output
        if args.visualizations_dir:
            metadata["visualizations_dir"] = args.visualizations_dir
        if args.feedback_note:
            metadata["note"] = args.feedback_note
        export_feedback(decisions, args.log_feedback, metadata=metadata)


if __name__ == "__main__":
    main()