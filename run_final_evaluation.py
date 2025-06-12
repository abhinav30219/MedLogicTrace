#!/usr/bin/env python3
"""
Final comprehensive evaluation script for MedLogicTrace project
Evaluates all components: baseline, GRPO, LogicTrace, and Multi-Agent systems
"""

import os
import torch
import json
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

from src.medical_evaluator import MedicalEvaluator
from src.data_utils import load_medical_datasets
from src.logictrace_grpo_trainer import LogicTraceGRPOTrainer
from src.multi_agent_medical_rl import MultiAgentMedicalSystem
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_baseline_model(
    model_name: str,
    test_data: List[Dict],
    device: str = "cuda"
) -> Dict:
    """Evaluate baseline model performance."""
    print(f"\nEvaluating baseline model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cuda":
        model = model.cuda()
    
    evaluator = MedicalEvaluator(model_name, device)
    results = []
    
    for item in tqdm(test_data, desc="Baseline evaluation"):
        prompt = f"Question: {item['question']}\n\nAnswer: "
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        eval_result = evaluator.evaluate_dataset(item['question'], response)
        eval_result['model'] = 'baseline'
        results.append(eval_result)
    
    return aggregate_results(results)


def evaluate_grpo_model(
    checkpoint_path: str,
    test_data: List[Dict],
    device: str = "cuda"
) -> Dict:
    """Evaluate GRPO-trained model."""
    print(f"\nEvaluating GRPO model: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get('model_name', 'Qwen/Qwen2.5-0.5B-Instruct')
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cuda":
        model = model.cuda()
    
    evaluator = MedicalEvaluator(tokenizer)
    results = []
    
    for item in tqdm(test_data, desc="GRPO evaluation"):
        prompt = f"Question: {item['question']}\n\nLet me think step by step.\n\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        eval_result = evaluator.evaluate_response(item['question'], response)
        eval_result['model'] = 'grpo'
        results.append(eval_result)
    
    return aggregate_results(results)


def evaluate_logictrace_model(
    checkpoint_path: str,
    test_data: List[Dict],
    device: str = "cuda"
) -> Dict:
    """Evaluate LogicTrace-optimized model."""
    print(f"\nEvaluating LogicTrace model: {checkpoint_path}")
    
    # Initialize LogicTrace trainer
    trainer = LogicTraceGRPOTrainer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        use_dapo_enhancements=True
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    evaluator = MedicalEvaluator(trainer.tokenizer)
    results = []
    
    for item in tqdm(test_data, desc="LogicTrace evaluation"):
        prompt = f"Question: {item['question']}\n\nLet me solve this step by step.\n\n"
        inputs = trainer.tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = trainer.model.generate(
                **inputs,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.95
            )
        
        response = trainer.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        eval_result = evaluator.evaluate_response(item['question'], response)
        eval_result['model'] = 'logictrace'
        
        # Analyze reasoning quality
        if 'answer' in item:
            reference = f"The answer is {item['answer']}"
            quality = trainer.logictrace_optimizer.analyze_reasoning_quality(
                response, reference
            )
            eval_result['token_efficiency'] = quality['token_efficiency_ratio']
            eval_result['step_compression'] = quality['step_compression_ratio']
        
        results.append(eval_result)
    
    return aggregate_results(results)


def evaluate_multiagent_system(
    model_name: str,
    test_data: List[Dict],
    device: str = "cuda"
) -> Dict:
    """Evaluate Multi-Agent medical system."""
    print(f"\nEvaluating Multi-Agent system")
    
    system = MultiAgentMedicalSystem(
        model_name=model_name,
        device=device,
        use_efficiency_agent=True
    )
    
    results = []
    
    for item in tqdm(test_data, desc="Multi-Agent evaluation"):
        # Process case through multi-agent system
        case_result = system.process_medical_case(
            item['question'],
            return_all_responses=True
        )
        
        # Extract answer from consensus
        answer = case_result['consensus_diagnosis']
        
        # Check correctness
        is_correct = False
        if 'answer' in item:
            is_correct = item['answer'].lower() in answer.lower()
        
        eval_result = {
            'is_correct': is_correct,
            'num_tokens': case_result['total_tokens_used'],
            'confidence': case_result['confidence_score'],
            'model': 'multiagent',
            'agent_tokens': {
                agent: resp.tokens_used 
                for agent, resp in case_result['agent_responses'].items() 
                if resp
            }
        }
        
        # Quality metrics
        quality = system.evaluate_consensus_quality(item['question'])
        eval_result.update(quality)
        
        results.append(eval_result)
    
    return aggregate_results(results)


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate evaluation results."""
    accuracy = np.mean([r['is_correct'] for r in results])
    avg_tokens = np.mean([r['num_tokens'] for r in results])
    
    # Calculate efficiency score
    efficiency_score = accuracy / (avg_tokens / 100) if avg_tokens > 0 else 0
    
    aggregated = {
        'accuracy': accuracy,
        'avg_tokens': avg_tokens,
        'efficiency_score': efficiency_score,
        'num_samples': len(results)
    }
    
    # Add additional metrics if available
    if 'confidence' in results[0]:
        aggregated['avg_confidence'] = np.mean([r['confidence'] for r in results])
    
    if 'token_efficiency' in results[0]:
        aggregated['token_efficiency'] = np.mean([r['token_efficiency'] for r in results if 'token_efficiency' in r])
    
    if 'agent_agreement' in results[0]:
        aggregated['agent_agreement'] = np.mean([r['agent_agreement'] for r in results])
    
    return aggregated


def create_comparison_plots(all_results: Dict[str, Dict], save_path: str):
    """Create comprehensive comparison plots."""
    models = list(all_results.keys())
    
    # Extract metrics
    accuracies = [all_results[m]['accuracy'] for m in models]
    tokens = [all_results[m]['avg_tokens'] for m in models]
    efficiency = [all_results[m]['efficiency_score'] for m in models]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('MedLogicTrace: Comprehensive Model Comparison', fontsize=16)
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy on Medical Reasoning')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom')
    
    # 2. Token usage
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, tokens, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax2.set_ylabel('Average Tokens per Response')
    ax2.set_title('Token Efficiency')
    
    for bar, tok in zip(bars2, tokens):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{tok:.0f}', ha='center', va='bottom')
    
    # 3. Efficiency score
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, efficiency, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax3.set_ylabel('Efficiency Score (Accuracy / Token Usage)')
    ax3.set_title('Overall Efficiency')
    
    for bar, eff in zip(bars3, efficiency):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{eff:.2f}', ha='center', va='bottom')
    
    # 4. Improvement summary
    ax4 = axes[1, 1]
    
    # Calculate improvements over baseline
    baseline_acc = all_results['Baseline']['accuracy']
    baseline_tokens = all_results['Baseline']['avg_tokens']
    
    improvements = []
    for model in models[1:]:  # Skip baseline
        acc_imp = (all_results[model]['accuracy'] - baseline_acc) / baseline_acc * 100
        tok_imp = (baseline_tokens - all_results[model]['avg_tokens']) / baseline_tokens * 100
        improvements.append({
            'model': model,
            'accuracy_improvement': acc_imp,
            'token_reduction': tok_imp
        })
    
    if improvements:
        imp_models = [imp['model'] for imp in improvements]
        acc_imps = [imp['accuracy_improvement'] for imp in improvements]
        tok_imps = [imp['token_reduction'] for imp in improvements]
        
        x = np.arange(len(imp_models))
        width = 0.35
        
        bars4_1 = ax4.bar(x - width/2, acc_imps, width, label='Accuracy Improvement (%)', color='#2ca02c')
        bars4_2 = ax4.bar(x + width/2, tok_imps, width, label='Token Reduction (%)', color='#9467bd')
        
        ax4.set_ylabel('Improvement %')
        ax4.set_title('Improvements over Baseline')
        ax4.set_xticks(x)
        ax4.set_xticklabels(imp_models)
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path}")


def generate_final_report(all_results: Dict[str, Dict], report_path: str):
    """Generate final evaluation report."""
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'summary': {
            'best_accuracy': max(all_results.items(), key=lambda x: x[1]['accuracy']),
            'best_efficiency': max(all_results.items(), key=lambda x: x[1]['efficiency_score']),
            'most_token_efficient': min(all_results.items(), key=lambda x: x[1]['avg_tokens'])
        },
        'detailed_results': all_results,
        'key_findings': []
    }
    
    # Add key findings
    if 'LogicTrace' in all_results:
        logictrace_imp = (all_results['LogicTrace']['accuracy'] - all_results['Baseline']['accuracy']) / all_results['Baseline']['accuracy'] * 100
        token_reduction = (all_results['Baseline']['avg_tokens'] - all_results['LogicTrace']['avg_tokens']) / all_results['Baseline']['avg_tokens'] * 100
        
        report['key_findings'].append(
            f"LogicTrace achieved {logictrace_imp:.1f}% accuracy improvement with {token_reduction:.1f}% token reduction"
        )
    
    if 'Multi-Agent' in all_results:
        report['key_findings'].append(
            f"Multi-Agent system achieved {all_results['Multi-Agent']['accuracy']:.1%} accuracy with consensus mechanisms"
        )
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nFinal report saved to {report_path}")
    print("\nKey Findings:")
    for finding in report['key_findings']:
        print(f"  - {finding}")


def main():
    """Run comprehensive final evaluation."""
    print("="*50)
    print("MedLogicTrace Final Evaluation")
    print("="*50)
    
    # Configuration
    config = {
        'test_datasets': ['medmcqa', 'pubmedqa'],
        'num_test_samples': 200,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'results_dir': 'results/final_evaluation'
    }
    
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Load test data
    print("\nLoading test datasets...")
    test_data = load_medical_datasets(config['test_datasets'], max_samples=config['num_test_samples'])
    print(f"Loaded {len(test_data)} test samples")
    
    # Results dictionary
    all_results = {}
    
    # 1. Evaluate baseline
    all_results['Baseline'] = evaluate_baseline_model(
        "Qwen/Qwen2.5-0.5B-Instruct",
        test_data,
        config['device']
    )
    
    # 2. Evaluate GRPO (if checkpoint exists)
    grpo_checkpoint = "models/grpo_custom_Qwen_Qwen2.5-0.5B-Instruct_20250526_062527/checkpoint_epoch_1/model.pt"
    if os.path.exists(grpo_checkpoint):
        all_results['GRPO'] = evaluate_grpo_model(
            grpo_checkpoint,
            test_data,
            config['device']
        )
    
    # 3. Evaluate LogicTrace (if checkpoint exists)
    logictrace_checkpoint = "models/logictrace_medical_20250603_140000/best_medical_model.pt"
    if os.path.exists(logictrace_checkpoint):
        all_results['LogicTrace'] = evaluate_logictrace_model(
            logictrace_checkpoint,
            test_data,
            config['device']
        )
    
    # 4. Evaluate Multi-Agent system
    all_results['Multi-Agent'] = evaluate_multiagent_system(
        "Qwen/Qwen2.5-0.5B-Instruct",
        test_data,
        config['device']
    )
    
    # Print results summary
    print("\n" + "="*50)
    print("Evaluation Results Summary")
    print("="*50)
    
    for model, results in all_results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {results['accuracy']:.1%}")
        print(f"  Avg Tokens: {results['avg_tokens']:.1f}")
        print(f"  Efficiency Score: {results['efficiency_score']:.2f}")
        
        if 'avg_confidence' in results:
            print(f"  Confidence: {results['avg_confidence']:.2f}")
        if 'token_efficiency' in results:
            print(f"  Token Efficiency: {results['token_efficiency']:.2f}")
        if 'agent_agreement' in results:
            print(f"  Agent Agreement: {results['agent_agreement']:.2f}")
    
    # Create comparison plots
    plot_path = os.path.join(config['results_dir'], 'final_comparison.png')
    create_comparison_plots(all_results, plot_path)
    
    # Generate final report
    report_path = os.path.join(config['results_dir'], 'final_report.json')
    generate_final_report(all_results, report_path)
    
    print("\n" + "="*50)
    print("Final evaluation completed!")
    print("="*50)


if __name__ == "__main__":
    main()
