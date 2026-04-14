# Laboratório 07 — Especialização de LLMs com LoRA e QLoRA

Fine-tuning eficiente de um modelo de linguagem fundacional usando PEFT/LoRA e quantização QLoRA, aplicado ao domínio de atendimento de barbearia, aproveitando para um projeto que estou desenvolvendo.

# **Nota de integridade:** Partes geradas/complementadas com IA, revisadas por Andre.

---
- Compatibilidade de código UTF-8 
- Prompt gerado por IA para otimizar o tempo e corrigir ortografia.
- método main de train.py e generate_dataset.py gerado por IA.
## Domínio escolhido

Assistente de atendimento da **Alpha Barber Club** — responde dúvidas de clientes via WhatsApp sobre serviços, preços, horários, agendamento e localização.

---
- Compatibilidade de código UTF-8 
- Prompt gerado por IA para otimizar o tempo e corrigir ortografia.
- método main de train.py e generate_dataset.py gerado por IA.
- 
## Estrutura do projeto

```
lab07-lora/
├── generate_dataset.py   # Gera o dataset sintético via API OpenAI
├── train.py              # Pipeline de fine-tuning com QLoRA
├── train.jsonl           # Dataset de treino (~90%)
├── test.jsonl            # Dataset de teste (~10%)
├── requirements.txt      # Dependências do projeto
└── README.md
```

> `.env`, `.venv/` e `adapter/` são ignorados pelo `.gitignore`.

---

## Passo 1 — Dataset sintético (`generate_dataset.py`)

- Usa a API OpenAI (`gpt-4o-mini`) para gerar exemplos realistas de atendimento
- 3 lotes de 20 exemplos cada, com variações temáticas
- Deduplicação automática antes do split
- Split 90/10 com seed fixo (`random.seed(42)`)
- Salva `train.jsonl` e `test.jsonl` no formato SFT:

```
### Instrução:
<mensagem do cliente>

### Resposta:
<resposta do atendente>
```

### Como gerar o dataset

```bash
# Configure a variável de ambiente
echo "OPENAI_API_KEY=sua_chave_aqui" > .env

pip install -r requirements.txt
python generate_dataset.py
```

---

## Passo 2 — Quantização QLoRA (`train.py`)

Modelo base carregado em **4-bit NF4** via `bitsandbytes`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
```

---

## Passo 3 — Configuração LoRA

Adaptadores treináveis com `peft.LoraConfig`:

| Hiperparâmetro | Valor |
|---|---|
| Rank (`r`) | 64 |
| Alpha (`lora_alpha`) | 16 |
| Dropout (`lora_dropout`) | 0.1 |
| Task type | `CAUSAL_LM` |

---

## Passo 4 — Pipeline de treinamento

Orquestrado com `SFTTrainer` da biblioteca `trl`.

| Parâmetro | Valor |
|---|---|
| Modelo base | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Otimizador | `paged_adamw_32bit` |
| LR Scheduler | `cosine` |
| Warmup ratio | `0.03` |
| Learning rate | `2e-4` |
| Épocas | `3` |
| Batch size | `1` (+ gradient accumulation 8) |
| Gradient checkpointing | `True` |
| Precisão | `fp16` |

### Como treinar

```bash
python train.py
```

O adapter é salvo em `./adapter/` ao final do treinamento.

---

## Dependências

```
openai>=1.0.0
torch>=2.0.0
transformers>=4.46.0
datasets>=2.14.0
peft>=0.14.0
trl>=0.12.0
bitsandbytes>=0.41.3
accelerate>=0.24.0
dotenv>=1.0.0
```

```bash
pip install -r requirements.txt
```

---

## Decisões técnicas

- **TinyLlama 1.1B** em vez do Llama 2 7B: viabiliza o treinamento local sem GPU de alto custo, mantendo toda a stack QLoRA/PEFT exigida.
- **`paged_adamw_32bit`**: transfere picos de memória do otimizador da GPU para a RAM, evitando OOM.
- **Scheduler cosine + warmup 3%**: estabiliza o início do treino e decai a LR suavemente, reduzindo risco de overfitting.
- **UTF-8 forçado** (`-X utf8` via `sys.execv`): garante compatibilidade em ambientes Windows sem configuração manual do terminal.
