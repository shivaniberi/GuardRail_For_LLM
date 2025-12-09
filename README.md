# Guardrail System for Phi-3-Mini LLM

This project provides a complete guardrail framework for the Phi-3-Mini language model. It introduces safety, reliability, and factual-accuracy checks around model inputs and outputs, ensuring more trustworthy and controlled deployments.

<div align="center"> <img width="593" height="383" alt="image" src="https://github.com/user-attachments/assets/3059c48d-7967-47c4-b55e-30dae6550604" /> </div>

<img width="2780" height="779" alt="image" src="https://github.com/user-attachments/assets/30fd90f1-b44f-482c-965a-ea46ae567ca0" />

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

```
User Prompt
   ↓
Input Guardrails (toxicity, safety, injection checks)
   ↓
Optional RAG Retrieval (context grounding)
   ↓
Phi-3-Mini Model (response generation)
   ↓
Output Guardrails (hallucination, confidence, toxicity, bias)
   ↓
Monitoring & Logging
```

---

## Setup

### Requirements

* Python 3.8+
* GPU recommended (optional)
* 16GB RAM recommended
* Local or mounted Phi-3-Mini model

### Installation

```bash
git clone <your-repo-url>
cd guardrail-system
pip install --break-system-packages -r requirements.txt
```

---

## Quick Start

```python
from phi3_guardrail_implementation import Phi3GuardrailSystem, GuardrailConfig

config = GuardrailConfig(
    phi3_model_path="/content/drive/MyDrive/phi-3-mini",
    confidence_threshold=0.7,
    toxicity_threshold=0.5,
    enable_rag=True
)

system = Phi3GuardrailSystem(config)

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

### RAG Retriever

Retrieves semantically relevant context to ground the model's response.

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
docker build -t phi3-guardrails .
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



