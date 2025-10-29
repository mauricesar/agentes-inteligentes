Repositório criado para registrar o conteúdo prático da disciplina de Agentes Inteligentes.

## Agente decisor para nódulos pulmonares

O diretório `lung_nodule_agent` reúne um agente que combina três detectores de nódulos em radiografias de tórax: YOLOv8, RT-DETR/DETR e Faster R-CNN. Cada adaptador roda a inferência do modelo correspondente, normaliza as saídas e o agente calcula uma fusão para emitir o veredito final.

### Instalação de dependências

```bash
pip install -r requirements.txt
```

### Como informar os modelos treinados

Os arquivos de pesos **não** precisam ficar dentro do repositório. Informe o caminho completo de cada arquivo `.pt` ou `.pth` no momento da execução. Duas abordagens são suportadas:

1. **Passar os caminhos na linha de comando.** Substitua os exemplos pelos locais onde você guardou seus checkpoints.

   ```bash
   python run_agent.py \
     --yolo /caminho/para/yolov8.pt \
     --detr /caminho/para/detr.pth \
     --faster-rcnn /caminho/para/faster_rcnn.pth \
     --image /caminho/para/imagem.png
   ```

   Para processar um diretório completo:

   ```bash
   python run_agent.py \
     --yolo /caminho/para/yolov8.pt \
     --detr /caminho/para/detr.pth \
     --faster-rcnn /caminho/para/faster_rcnn.pth \
     --dataset /caminho/para/pasta_de_testes \
     --output resultados.json \
     --log-feedback feedbacks.jsonl \
     --feedback-note "Execução em 05/06/2024"
   ```

2. **Criar um script de configuração próprio.** Ele centraliza os caminhos dos pesos para que você não precise digitá-los sempre.

   ```python
   from lung_nodule_agent import YOLOv8Adapter, DETRAdapter, FasterRCNNAdapter, LungNoduleDecisionAgent

   adapters = [
       YOLOv8Adapter("/caminho/para/yolov8.pt"),
       DETRAdapter("/caminho/para/detr.pth"),
       FasterRCNNAdapter("/caminho/para/faster_rcnn.pth"),
   ]
   agent = LungNoduleDecisionAgent(adapters)
   resultado = agent.predict("/caminho/para/imagem.png")
   print(resultado.to_dict())
   ```

   Salve o exemplo como `meu_agente.py` e execute com `python meu_agente.py` depois de ajustar os caminhos.

### Salvando evidências visuais

O agente pode gerar imagens anotadas para cada modelo e para a fusão. Existem duas formas de habilitar o recurso:

1. **Usar o parâmetro `--visualizations-dir` na linha de comando.** O script cria subpastas chamadas `YOLOv8`, `DETR`, `Faster R-CNN` e `fusion` dentro do diretório informado. Cada imagem recebe as caixas previstas pelo modelo correspondente.

   ```bash
   python run_agent.py \
     --yolo /caminho/para/yolov8.pt \
     --detr /caminho/para/detr.pth \
     --faster-rcnn /caminho/para/faster_rcnn.pth \
     --dataset /caminho/para/pasta_de_testes \
     --visualizations-dir saidas_visuais
   ```

2. **Ativar o diretório no script `meu_agente.py`.** Ajuste a constante `VISUALIZATIONS_DIR` para apontar para a pasta desejada ou defina como `None` se quiser desativar a exportação.

### Registrando feedbacks e resultados

A flag `--log-feedback` cria ou atualiza um arquivo **JSON Lines** (`.jsonl`) com todas as decisões do agente, incluindo parâmetros e observações opcionais (`--feedback-note`). Cada linha guarda uma execução e pode ser versionada no GitHub.

Também é possível registrar feedback programaticamente:

```python
from lung_nodule_agent import export_feedback

decisions = agent.evaluate_directory("/caminho/pasta_testes")
export_feedback(
    decisions,
    "feedbacks.jsonl",
    metadata={"note": "Rodada de validação local"},
)
```

### Ajustando parâmetros principais

* `--confidence-threshold` define o limiar mínimo para aceitar uma detecção individual (padrão 0.25).
* `--vote-threshold` define quantos modelos precisam concordar para emitir "Provável presença de nódulo".
* `--iou-threshold` define o limite de IoU usado para mesclar detecções (padrão 0.5).

Os resultados ficam disponíveis em JSON (arquivo indicado em `--output` ou saída padrão) e trazem as detecções de cada modelo e as detecções agregadas.

### Compatibilidade com TorchVision e Ultralytics

Algumas versões antigas do TorchVision expõem construtores diferentes. Se você receber mensagens como `cannot import name 'detr_resnet50'` ou erros envolvendo parâmetros `weights` ou `pretrained`, atualize para **torchvision 0.13 ou superior**. O adaptador tenta instanciar o modelo via TorchVision e, caso não consiga, recorre automaticamente ao `torch.hub` (`facebookresearch/detr`). Na primeira execução pode ocorrer um download com cache em `~/.cache/torch/hub`.

Checkpoints do **Ultralytics RT-DETR** também são aceitos. Quando o adaptador identifica esse formato, ele usa o runtime da Ultralytics. Certifique-se de instalar o pacote `ultralytics` antes de usar pesos RT-DETR.

> ℹ️ **PyTorch 2.6+ e checkpoints customizados**
>
> A versão 2.6 do PyTorch passou a usar `weights_only=True` por padrão em `torch.load`, bloqueando classes externas durante a desserialização. Os adaptadores registram automaticamente as classes necessárias e tentam nova carga em modo seguro. Como último recurso (apenas se você confiar no arquivo) o carregamento altera para `weights_only=False` para aceitar checkpoints legados.