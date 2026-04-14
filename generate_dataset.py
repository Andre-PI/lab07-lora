import json
import random
import time
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SALON_INFO = """
Barbearia: Alpha Barber Club
Horário: Seg-Sex 9h-20h | Sáb 9h-18h | Dom fechado
Endereço: Av. Dom Severino, 456, Teresina - PI
WhatsApp/Tel: (86) 98888-4321
Pagamento: Pix, débito, crédito, dinheiro

Serviços e preços:
- Corte masculino: R$40
- Corte + barba: R$60
- Barba completa: R$30
- Degradê (fade): R$45
- Navalhado: R$50
- Pigmentação de barba: R$35
- Hidratação capilar: R$50
- Sobrancelha masculina: R$15

Agendamento: apenas pelo WhatsApp ou telefone.
Atendimento por ordem de chegada disponível em horários de menor movimento.
Estacionamento: disponível na rua.

Observações:
- Tempo médio corte: 30 minutos
- Corte + barba: 50 minutos
- Trabalhamos com produtos profissionais para cabelo e barba
"""

BASE_PROMPT = """Você está gerando dados de treinamento para um assistente de atendimento de barbearia masculina.

Informações da barbearia:
{SALON_INFO}

Gere {n} exemplos REALISTAS e variados de atendimento entre cliente e barbearia.

Regras IMPORTANTES:
- As perguntas devem parecer mensagens reais de WhatsApp
- O cliente pode escrever de forma curta, informal, com abreviações e pequenos erros
- As respostas devem ser claras, diretas, educadas e naturais
- Use SOMENTE as informações fornecidas
- NÃO invente preço, horário, serviço, promoção, profissional ou política de atendimento
- Evite repetições e exemplos quase iguais
- Gere exemplos úteis para treinamento supervisionado

Cubra bem estes temas:
- horários de funcionamento
- preços e serviços
- agendamento
- tempo de duração
- formas de pagamento
- localização
- estacionamento
- ordem de chegada
- diferença entre serviços
- cuidados após cabelo/barba
- dúvidas sobre produtos usados
- perguntas sobre disponibilidade do dia

Estilo:
- Cliente: linguagem natural de WhatsApp
- Atendente: humano, objetivo e profissional
- Respostas curtas ou médias, sem enrolação

Retorne em JSON válido no formato:
{{
  "examples": [
    {{
      "instruction": "mensagem do cliente",
      "response": "resposta do atendente"
    }}
  ]
}}
"""

INSTRUCTIONS = [
    "Dê prioridade a perguntas curtas e comuns.",
    "Inclua mais perguntas sobre barba, degradê e corte + barba.",
    "Inclua mais perguntas sobre horários, pagamento e localização.",
    "Inclua mais exemplos com dúvidas realistas de primeira visita.",
    "Inclua algumas mensagens com erros leves de digitação.",
]


def buildPrompt(n: int, variation: str) -> str:
    return BASE_PROMPT.format(
        SALON_INFO=SALON_INFO,
        n=n
    ) + f"\nInstrução extra desta rodada: {variation}\n"


def parseExamples(content: str) -> List[Dict[str, str]]:
    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("A resposta JSON não é um objeto.")

    if "examples" not in data:
        raise ValueError("Campo 'examples' não encontrado.")

    examples = data["examples"]
    if not isinstance(examples, list):
        raise ValueError("'examples' não é uma lista.")

    cleaned = []
    for item in examples:
        if not isinstance(item, dict):
            continue

        instruction = str(item.get("instruction", "")).strip()
        response = str(item.get("response", "")).strip()

        if not instruction or not response:
            continue

        cleaned.append({
            "instruction": instruction,
            "response": response
        })

    if not cleaned:
        raise ValueError("Nenhum exemplo válido encontrado.")

    return cleaned


def generateBatch(n: int, variation: str, maxRetries: int = 3) -> List[Dict[str, str]]:
    prompt = buildPrompt(n, variation)

    for attempt in range(1, maxRetries + 1):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Você gera datasets sintéticos de alta qualidade. Responda apenas em JSON válido."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,
                response_format={"type": "json_object"}
            )

            content = resp.choices[0].message.content
            if content is None:
                raise ValueError("A resposta da API veio vazia.")

            return parseExamples(content)

        except Exception as e:
            if attempt == maxRetries:
                raise RuntimeError(
                    f"Falha ao gerar lote após {maxRetries} tentativas: {e}"
                ) from e
            time.sleep(1.5 * attempt)

    return []


def deduplicateExamples(examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique = []

    for ex in examples:
        key = (
            ex["instruction"].strip().lower(),
            ex["response"].strip().lower()
        )
        if key not in seen:
            seen.add(key)
            unique.append(ex)

    return unique


def toSftFormat(examples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [
        {
            "text": f"### Instrução:\n{ex['instruction']}\n\n### Resposta:\n{ex['response']}"
        }
        for ex in examples
    ]


def saveJsonl(fileName: str, data: List[Dict[str, str]]) -> None:
    with open(fileName, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    print("CALLING OpenAI...")

    allExamples = []
    batchPlan = [20, 20, 20]

    for i, batchSize in enumerate(batchPlan, start=1):
        variation = INSTRUCTIONS[(i - 1) % len(INSTRUCTIONS)]
        print(f"  Lote {i} ({batchSize} exemplos)...")

        batch = generateBatch(batchSize, variation)
        print(f"  Retornaram {len(batch)} exemplos")

        allExamples.extend(batch)
        allExamples = deduplicateExamples(allExamples)

        print(f"  Acumulado único: {len(allExamples)} exemplos")

    formatted = toSftFormat(allExamples)

    random.seed(42)
    random.shuffle(formatted)

    split = int(len(formatted) * 0.9)
    trainData = formatted[:split]
    testData = formatted[split:]

    saveJsonl("train.jsonl", trainData)
    saveJsonl("test.jsonl", testData)

    print("\nConcluído.")
    print(f"Total único: {len(formatted)} exemplos")
    print(f"Treino: {len(trainData)}")
    print(f"Teste: {len(testData)}")
    print("Arquivos salvos: train.jsonl, test.jsonl")


if __name__ == "__main__":
    main()