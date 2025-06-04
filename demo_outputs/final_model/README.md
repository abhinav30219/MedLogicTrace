---
license: apache-2.0
tags:
- demo
- medlogictrace
---

# MedLogicTrace Demo Model

This is a demo model created by the MedLogicTrace training pipeline.

## Model Details
- Base Model: Qwen/Qwen2.5-0.5B
- Training: Simple demo with 5 examples
- Purpose: Demonstration of the complete pipeline

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("abhinav302019/medlogictrace-demo-20250603_154459")
tokenizer = AutoTokenizer.from_pretrained("abhinav302019/medlogictrace-demo-20250603_154459")

prompt = "Question: What is 2 + 2?\nAnswer:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Training Pipeline

This model was created using the MedLogicTrace training pipeline, which includes:
- Mathematical pretraining on Bespoke-Stratos-17k
- Medical fine-tuning on MedMCQA + PubMedQA
- Stable GRPO training with numerical stability fixes

For the full training pipeline, see the MedLogicTrace repository.
