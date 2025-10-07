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

### Onde informar os modelos treinados

Os pesos dos modelos **não** precisam ser copiados para dentro do repositório. Basta informar o caminho
para cada arquivo `.pt`/`.pth` na hora de executar o script de inferência. Você pode fazer isso de duas formas:

1. **Passando os argumentos diretamente na linha de comando**, como nos exemplos abaixo. Substitua os caminhos
   pelos locais onde seus arquivos realmente estão salvos (disco local, pendrive, volume montado, etc.).

   ```bash
   python run_agent.py \
     --yolo /caminho/para/yolov8.pt \
     --detr /caminho/para/detr.pth \
     --faster-rcnn /caminho/para/faster_rcnn.pth \
     --image /caminho/para/imagem.png
   ```

   Para avaliar um diretório inteiro:

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

2. **Criando um pequeno script de configuração próprio** (opcional) onde você instancia o agente e informa os
   caminhos apenas uma vez. Exemplo:

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

   Salve esse exemplo em um arquivo (por exemplo, `meu_agente.py`) e execute com `python meu_agente.py`. Assim,
   você concentra os caminhos dos modelos em um único lugar caso não queira digitá-los sempre.

### Registrando feedbacks e resultados para versionamento

Para guardar o histórico das execuções (e fazer upload no GitHub, se desejar), utilize a opção `--log-feedback`.
Ela cria ou atualiza um arquivo no formato **JSON Lines** (`.jsonl`) com todas as decisões do agente, incluindo
os caminhos dos modelos, parâmetros e uma anotação opcional (`--feedback-note`). Cada linha representa uma
execução e pode ser facilmente versionada.

Você também pode chamar programaticamente a função `export_feedback`:

```python
from lung_nodule_agent import export_feedback

decisions = agent.evaluate_directory("/caminho/pasta_testes")
export_feedback(
    decisions,
    "feedbacks.jsonl",
    metadata={"note": "Rodada de validação local"},
)
```

Parâmetros adicionais do CLI:

- `--confidence-threshold`: limiar mínimo de confiança para considerar uma detecção individual (padrão 0.25).
- `--vote-threshold`: número mínimo de modelos que devem concordar para o veredito "Provável presença de nódulo".
- `--iou-threshold`: limite de IoU utilizado para mesclar detecções entre modelos (padrão 0.5).

O resultado é salvo em JSON (arquivo especificado em `--output` ou impresso no terminal), incluindo as
detecções individuais de cada modelo e as detecções agregadas pelo agente.

#### Compatibilidade com versões do TorchVision

Algumas versões mais antigas do TorchVision não expõem as mesmas assinaturas dos construtores usados
pelos adaptadores. Caso receba erros do tipo `cannot import name 'detr_resnet50'` ou mensagens indicando
parâmetros desconhecidos (`weights`/`pretrained`), atualize para **torchvision 0.13 ou superior**. As
últimas alterações no agente tentam detectar automaticamente essas diferenças, mas, se o pacote estiver
muito desatualizado, é recomendado atualizar (`pip install --upgrade torchvision`) ou fornecer seu próprio
construtor ao estender os adaptadores.
