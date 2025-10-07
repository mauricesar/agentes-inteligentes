# agentes-inteligentes

Repositório criado para armazenar o progresso do conteúdo prático da disciplina de Agentes Inteligentes.

## Agente decisor para nódulos pulmonares

O diretório `lung_nodule_agent` contém um agente capaz de combinar três modelos de detecção de nódulos
pulmonares (YOLOv8, DETR e Faster R-CNN) previamente treinados. O agente realiza a inferência com cada
modelo, funde as detecções e emite um veredito indicando a presença ou não de regiões suspeitas.

### Instalação de dependências

```bash
pip install -r requirements.txt
```

### Utilização via linha de comando

```bash
python run_agent.py \
  --yolo /caminho/para/yolov8.pt \
  --detr /caminho/para/detr.pth \
  --faster-rcnn /caminho/para/faster_rcnn.pth \
  --image /caminho/para/imagem.png
```

Também é possível executar o agente em um diretório com múltiplas imagens usando a flag `--dataset`:

```bash
python run_agent.py \
  --yolo /caminho/para/yolov8.pt \
  --detr /caminho/para/detr.pth \
  --faster-rcnn /caminho/para/faster_rcnn.pth \
  --dataset /caminho/para/pasta_de_testes \
  --output resultados.json
```

Parâmetros adicionais:

- `--confidence-threshold`: limiar mínimo de confiança para considerar uma detecção individual (padrão 0.25).
- `--vote-threshold`: número mínimo de modelos que devem concordar para o veredito "Provável presença de nódulo".
- `--iou-threshold`: limite de IoU utilizado para mesclar detecções entre modelos (padrão 0.5).

O resultado é salvo em JSON (arquivo especificado em `--output` ou impresso no terminal), incluindo as
detecções individuais de cada modelo e as detecções agregadas pelo agente.
