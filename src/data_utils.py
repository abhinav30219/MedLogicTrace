"""Data loading utilities for MedLogicTrace"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple
import random
from tqdm import tqdm


def load_math_dataset(dataset_name: str, subset_size: int = 10000) -> Dict:
    """Load and prepare mathematical reasoning dataset for GRPO training"""
    print(f"Loading {dataset_name}...")
    
    # Load the dataset
    dataset = load_dataset(dataset_name, split="train")
    
    # Take a subset if specified
    if subset_size and subset_size < len(dataset):
        indices = random.sample(range(len(dataset)), subset_size)
        dataset = dataset.select(indices)
    
    print(f"Loaded {len(dataset)} examples from {dataset_name}")
    return dataset


def prepare_grpo_data(dataset, tokenizer, max_length: int = 512) -> List[Dict]:
    """Prepare data in the format expected by TRL's GRPOTrainer"""
    prepared_data = []
    
    for example in tqdm(dataset, desc="Preparing GRPO data"):
        # Format the prompt
        if "question" in example:
            prompt = f"Question: {example['question']}\nAnswer:"
        elif "problem" in example:
            prompt = f"Problem: {example['problem']}\nSolution:"
        else:
            # Skip if no recognizable field
            continue
            
        prepared_data.append({
            "query": prompt,
            "input_ids": tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        })
    
    return prepared_data


def load_medical_dataset(dataset_name: str, subset_size: int = 1000) -> List[Dict]:
    """Load medical evaluation datasets"""
    print(f"Loading {dataset_name}...")
    
    if dataset_name == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
        # Format: question with multiple choice
        examples = []
        for item in dataset:
            if len(examples) >= subset_size:
                break
            question = item["question"]
            options = [item["opa"], item["opb"], item["opc"], item["opd"]]
            correct_idx = item["cop"]
            
            examples.append({
                "question": question,
                "options": options,
                "answer": correct_idx,
                "full_question": f"{question}\nA) {options[0]}\nB) {options[1]}\nC) {options[2]}\nD) {options[3]}"
            })
    
    elif dataset_name == "medqa":
        # MedQA USMLE dataset
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
        examples = []
        for item in dataset:
            if len(examples) >= subset_size:
                break
            examples.append({
                "question": item["question"],
                "options": [item["options"]["A"], item["options"]["B"], 
                           item["options"]["C"], item["options"]["D"]],
                "answer": ord(item["answer"]) - ord('A'),  # Convert A/B/C/D to 0/1/2/3
                "full_question": item["question"]
            })
    
    elif dataset_name == "pubmed_qa":
        dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        examples = []
        for item in dataset:
            if len(examples) >= subset_size:
                break
            # PubMedQA is yes/no/maybe format
            question = item["question"]
            context = item["context"]["contexts"][0] if item["context"]["contexts"] else ""
            answer = item["final_decision"]
            
            examples.append({
                "question": f"{question}\nContext: {context[:500]}...",  # Truncate context
                "options": ["yes", "no", "maybe"],
                "answer": ["yes", "no", "maybe"].index(answer),
                "full_question": f"{question}\nContext: {context[:500]}..."
            })
    
    print(f"Loaded {len(examples)} examples from {dataset_name}")
    return examples


def format_medical_prompt(example: Dict, include_answer: bool = False) -> str:
    """Format medical question for model input"""
    prompt = f"Medical Question: {example['full_question']}\n\n"
    
    if "options" in example:
        prompt += "Options:\n"
        for i, option in enumerate(example['options']):
            prompt += f"{chr(65+i)}) {option}\n"
    
    prompt += "\nAnswer:"
    
    if include_answer:
        if "options" in example:
            prompt += f" {chr(65 + example['answer'])}"
        else:
            prompt += f" {example['answer']}"
    
    return prompt


def evaluate_medical_response(response: str, example: Dict) -> bool:
    """Check if model response is correct"""
    response = response.strip().lower()
    
    if "options" in example:
        # Multiple choice - check if response contains correct letter
        correct_letter = chr(65 + example['answer']).lower()
        return correct_letter in response[:10]  # Check first 10 chars
    else:
        # Direct answer
        return str(example['answer']).lower() in response


def calculate_token_efficiency(responses: List[str], tokenizer) -> Dict[str, float]:
    """Calculate token efficiency metrics"""
    token_counts = []
    
    for response in responses:
        tokens = tokenizer.encode(response)
        token_counts.append(len(tokens))
    
    return {
        "avg_tokens": sum(token_counts) / len(token_counts),
        "min_tokens": min(token_counts),
        "max_tokens": max(token_counts),
        "total_tokens": sum(token_counts)
    }
