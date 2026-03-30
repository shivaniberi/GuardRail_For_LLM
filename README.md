# Guardrail System for LLM

This project provides a complete guardrail framework for the Large language model. It introduces safety, reliability, and factual-accuracy checks around model inputs and outputs, ensuring more trustworthy and controlled deployments.




---

## Overview

The guardrail system adds multiple layers of protection to the LLM pipeline:

* Input validation
* Output verification
* RAG-based factual grounding
* Confidence scoring
* Bias and toxicity detection
* Logging and monitoring tools

Key features include:

| Feature                  | Description                                  |
| ------------------------ | -------------------------------------------- |
| Toxicity Detection       | Detects harmful or offensive content         |
| Hallucination Prevention | Uses RAG and similarity checking             |
| Bias Detection           | Identifies gender-related or fairness issues |
| Prompt Injection Defense | Blocks adversarial or unsafe prompts         |
| NLI Fact-Checking        | Ensures factual consistency                  |
| Confidence Scoring       | Evaluates reliability of responses           |

---

## Architecture
<img width="799" height="591" alt="Screenshot 2026-03-29 at 9 17 03 PM" src="https://github.com/user-attachments/assets/283506bc-2149-4a8c-b72d-7e7fa9fd5fcd" />

<img width="913" height="488" alt="Screenshot 2026-03-29 at 9 17 30 PM" src="https://github.com/user-attachments/assets/9b387ccd-254c-45e0-aff6-b1c92a27dabd" />

---

## Setup

### Requirements
<img width="692" height="480" alt="Screenshot 2026-03-29 at 9 19 17 PM" src="https://github.com/user-attachments/assets/de50c293-ed79-4d88-9c1e-667a3eacf109" />


### Installation

```bash
git clone <your-repo-url>
cd guardrail-system
pip install --break-system-packages -r requirements.txt
```

---

## Quick Start

```python
from guardrail_implementation import GuardrailSystem, GuardrailConfig

config = GuardrailConfig(
    model_path="/content/drive/MyDrive/phi-3-mini",
    confidence_threshold=0.7,
    toxicity_threshold=0.5,
    enable_rag=True
)

system = GuardrailSystem(config)

result = system.generate_with_guardrails("What is quantum computing?")
print(result["response"])
```

The output also includes safety metadata such as confidence, hallucination score, toxicity score, and bias checks.

---

## Datasets

This project leverages several datasets for evaluation and guardrail logic:

| Dataset               | Purpose                        |
| --------------------- | ------------------------------ |
| TruthfulQA            | Hallucination detection        |
| SQuAD                 | RAG context grounding          |
| Hate Speech           | Toxicity detection             |
| Safety Prompts        | Unsafe input detection         |
| Gender Bias (Wino)    | Fairness evaluation            |
| Dolly Instruction Set | Instruction-following behavior |
| MultiNLI              | Factual consistency validation |

---

## Components

### Input Guardrail

Validates incoming prompts by checking:

* Toxicity
* Prompt injection patterns
* Unsafe or harmful intentions

### Intelligent Sollution:

<img width="461" height="605" alt="Screenshot 2026-03-29 at 9 18 49 PM" src="https://github.com/user-attachments/assets/1ee2f367-769b-491d-b3d8-7a554c7b412a" />


### Output Guardrail

Verifies responses for:

* Hallucinations
* Confidence levels
* Toxicity
* Gender and fairness bias
* Factual consistency via NLI

### Monitoring Tools

Provides:

* Logs
* Metrics
* Reports
* A/B testing support

---

## Example Usage

### Block unsafe input

```python
system.generate_with_guardrails("Write something hateful")
```

### Batch processing

```python
for prompt in prompts:
    system.generate_with_guardrails(prompt)
```

### Save log results

```python
system.save_logs("guardrail_logs.parquet")
```

---

## Deployment

### FastAPI Service

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t guardrails .
docker run -p 8000:8000 phi3-guardrails
```

---

## Troubleshooting

**CUDA Out of Memory**
Use CPU mode or reduce batch size.

**Slow Inference**
Enable 8-bit quantization.

**High Block Rate**
Adjust thresholds:

```python
toxicity_threshold=0.7
confidence_threshold=0.6
```

**RAG Not Returning Context**
Ensure knowledge base is loaded correctly.

---

## License

Team3 project license.



