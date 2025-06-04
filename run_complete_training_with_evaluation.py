#!/usr/bin/env python3
"""
Complete MedLogicTrace training pipeline with comprehensive evaluation
- Trains on Bespoke-Stratos-17k and medical data
- Evaluates all 3 models (base, math, final) on benchmarks
- Uses multi-agent RL for enhanced evaluation
- Automatically uploads everything to HuggingFace
"""

import os
import sys
import torch
import json
import logging
from datetime import datetime
from huggingface_hub import HfApi, create_repo
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.stable_grpo_trainer_accelerate import StableGRPOTrainerAccelerate as StableGRPOTrainer
from src.math_data_utils import MathDatasetLoader
from src.data_utils import load_medical_datasets
from src.medical_evaluator import MedicalEvaluator
from src.multi_agent_medical_rl import MultiAgentMedicalSystem
from fix_gpu_stability import apply_all_stability_fixes
from run_runpod_training_with_hf import (
    HuggingFaceUploader, 
    create_model_card,
    export_tensorboard_to_images
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    # Model settings
    'base_model_name': 'Qwen/Qwen2.5-0.5B-Instruct',
    
    # Dataset settings
    'math_dataset': 'bespoke-stratos',
    'math_samples': 1000,
    'medical_samples': 5000,
    
    # Training settings
    'batch_size': 4,
    'learning_rate_math': 5e-6,
    'learning_rate_medical': 2e-6,
    'num_epochs_math': 3,
    'num_epochs_medical': 3,
    
    # GRPO settings
    'k_samples': 2,
    'temperature': 1.0,
    'kl_coef': 0.2,
    'max_grad_norm': 0.5,
    'min_temperature': 0.5,
    
    # Evaluation settings
    'eval_batch_size': 8,
    'benchmark_samples': 500,  # Per benchmark
    'use_multi_agent': True,
    
    # HuggingFace settings
    'hf_username': 'abhinav302019',
    'hf_token': os.environ.get('HUGGINGFACE_TOKEN'),
    'repo_base_name': 'medlogictrace-stratos',
    
    # Directories
    'checkpoint_dir': 'checkpoints',
    'tensorboard_dir': 'tensorboard_logs',
    'results_dir': 'results',
}

class ComprehensiveEvaluator:
    """Evaluate all models on multiple benchmarks with multi-agent RL."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.benchmarks = ['medmcqa', 'medqa', 'pubmed_qa']
        
    def evaluate_all_models(
        self,
        base_model_path: str,
        math_model_path: str,
        final_model_path: str,
        tb_writer: SummaryWriter
    ) -> Dict:
        """Evaluate all 3 models on benchmarks."""
        logger.info("="*60)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*60)
        
        models_to_evaluate = {
            'base': base_model_path,
            'math': math_model_path,
            'final': final_model_path
        }
        
        all_results = {}
        
        # Single-model evaluation
        logger.info("\n1. Single-Model Evaluation")
        logger.info("-"*40)
        
        for model_name, model_path in models_to_evaluate.items():
            logger.info(f"\nEvaluating {model_name} model: {model_path}")
            
            # Load evaluator
            evaluator = MedicalEvaluator(model_path, self.device)
            
            model_results = {}
            
            # Evaluate on each benchmark
            for benchmark in self.benchmarks:
                logger.info(f"  - Benchmark: {benchmark}")
                
                # Load test data
                test_data = load_medical_datasets(
                    [benchmark],
                    max_samples=self.config['benchmark_samples']
                )
                
                # Evaluate
                results = evaluator.evaluate_dataset(
                    benchmark,
                    subset_size=len(test_data)
                )
                
                model_results[benchmark] = results
                
                # Log to TensorBoard
                tb_writer.add_scalar(
                    f'Evaluation/{model_name}/{benchmark}/Accuracy',
                    results['accuracy'],
                    0
                )
                tb_writer.add_scalar(
                    f'Evaluation/{model_name}/{benchmark}/AvgTokens',
                    results['avg_tokens'],
                    0
                )
            
            all_results[model_name] = model_results
            
            # Clean up
            del evaluator
            torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        # Multi-agent evaluation
        if self.config['use_multi_agent']:
            logger.info("\n2. Multi-Agent RL Evaluation")
            logger.info("-"*40)
            
            multi_agent_results = self._evaluate_with_multi_agent(
                models_to_evaluate,
                tb_writer
            )
            
            all_results['multi_agent'] = multi_agent_results
        
        # Generate comparison visualizations
        self._generate_comparison_plots(all_results)
        
        return all_results
    
    def _evaluate_with_multi_agent(
        self,
        models: Dict[str, str],
        tb_writer: SummaryWriter
    ) -> Dict:
        """Evaluate models using multi-agent system."""
        multi_agent_results = {}
        
        for model_name, model_path in models.items():
            logger.info(f"\nMulti-agent evaluation for {model_name} model")
            
            # Initialize multi-agent system
            mas = MultiAgentMedicalSystem(
                model_name=model_path,
                device=self.device,
                use_efficiency_agent=True
            )
            
            # Test on subset of medical cases
            test_cases = load_medical_datasets(
                ['medmcqa'],
                max_samples=100  # Smaller subset for multi-agent
            )
            
            accuracies = []
            token_savings = []
            consensus_qualities = []
            
            for case in tqdm(test_cases, desc=f"Multi-agent eval ({model_name})"):
                # Format case
                case_description = f"Question: {case['question']}"
                
                # Process through multi-agent system
                result = mas.process_medical_case(
                    case_description,
                    return_all_responses=True
                )
                
                # Evaluate quality
                quality = mas.evaluate_consensus_quality(
                    case_description,
                    ground_truth={'diagnosis': str(case.get('answer', ''))}
                )
                
                # Check accuracy
                predicted_answer = result['consensus_diagnosis']
                is_correct = str(case.get('answer', '')).lower() in predicted_answer.lower()
                accuracies.append(float(is_correct))
                
                # Calculate token savings
                single_model_tokens = result['total_tokens_used'] / 4  # Approximate
                efficient_tokens = len(result.get('efficient_summary', '').split())
                savings = 1.0 - (efficient_tokens / single_model_tokens) if single_model_tokens > 0 else 0
                token_savings.append(savings)
                
                # Consensus quality
                consensus_qualities.append(quality['agent_agreement'])
            
            # Aggregate results
            multi_agent_results[model_name] = {
                'accuracy': np.mean(accuracies),
                'token_savings': np.mean(token_savings),
                'consensus_quality': np.mean(consensus_qualities),
                'agreement_variance': np.std(consensus_qualities)
            }
            
            # Log to TensorBoard
            for metric, value in multi_agent_results[model_name].items():
                tb_writer.add_scalar(
                    f'MultiAgent/{model_name}/{metric}',
                    value,
                    0
                )
            
            # Clean up
            del mas
            torch.cuda.empty_cache() if self.device == 'cuda' else None
        
        return multi_agent_results
    
    def _generate_comparison_plots(self, results: Dict):
        """Generate comprehensive comparison plots."""
        logger.info("\nGenerating comparison plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MedLogicTrace: Comprehensive Model Evaluation', fontsize=16)
        
        # 1. Accuracy comparison across benchmarks
        ax = axes[0, 0]
        benchmarks = list(results['base'].keys())
        x = np.arange(len(benchmarks))
        width = 0.25
        
        for i, (model_name, model_results) in enumerate(['base', 'math', 'final']):
            if model_name in results:
                accuracies = [results[model_name][b]['accuracy'] for b in benchmarks]
                ax.bar(x + i*width, accuracies, width, label=model_name.capitalize())
        
        ax.set_xlabel('Benchmark')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Across Benchmarks')
        ax.set_xticks(x + width)
        ax.set_xticklabels(benchmarks)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Token efficiency comparison
        ax = axes[0, 1]
        models = ['base', 'math', 'final']
        avg_tokens = []
        for model in models:
            if model in results:
                tokens = [results[model][b]['avg_tokens'] for b in benchmarks]
                avg_tokens.append(np.mean(tokens))
        
        ax.bar(models, avg_tokens, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_xlabel('Model')
        ax.set_ylabel('Average Tokens')
        ax.set_title('Token Efficiency')
        ax.grid(True, alpha=0.3)
        
        # 3. Multi-agent performance
        if 'multi_agent' in results:
            ax = axes[0, 2]
            ma_results = results['multi_agent']
            
            metrics = ['accuracy', 'token_savings', 'consensus_quality']
            model_names = list(ma_results.keys())
            
            x = np.arange(len(model_names))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [ma_results[m][metric] for m in model_names]
                ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Multi-Agent RL Performance')
            ax.set_xticks(x + width)
            ax.set_xticklabels(model_names)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. Overall performance radar chart
        ax = axes[1, 0]
        ax.remove()
        ax = fig.add_subplot(2, 3, 4, projection='polar')
        
        categories = ['Accuracy', 'Efficiency', 'Consensus', 'Speed']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        for model in ['base', 'math', 'final']:
            if model in results:
                # Normalize metrics to 0-1 scale
                accuracy = np.mean([results[model][b]['accuracy'] for b in benchmarks])
                efficiency = 1.0 - (np.mean([results[model][b]['avg_tokens'] for b in benchmarks]) / 200)
                consensus = results.get('multi_agent', {}).get(model, {}).get('consensus_quality', 0.5)
                speed = 1.0 - (np.mean([results[model][b]['avg_tokens'] for b in benchmarks]) / 300)
                
                values = [accuracy, efficiency, consensus, speed]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model.capitalize())
                ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Overall Performance Comparison', y=1.08)
        ax.legend()
        ax.grid(True)
        
        # 5. Training progression (placeholder)
        ax = axes[1, 1]
        ax.text(0.5, 0.5, 'Training Loss Curves\n(Generated during training)', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Training Progression')
        ax.axis('off')
        
        # 6. Model comparison table
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        # Create comparison data
        table_data = []
        for model in ['base', 'math', 'final']:
            if model in results:
                avg_acc = np.mean([results[model][b]['accuracy'] for b in benchmarks])
                avg_tokens = np.mean([results[model][b]['avg_tokens'] for b in benchmarks])
                ma_quality = results.get('multi_agent', {}).get(model, {}).get('consensus_quality', 'N/A')
                
                table_data.append([
                    model.capitalize(),
                    f"{avg_acc:.2%}",
                    f"{avg_tokens:.0f}",
                    f"{ma_quality:.2f}" if isinstance(ma_quality, float) else ma_quality
                ])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Model', 'Avg Accuracy', 'Avg Tokens', 'MA Quality'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title('Model Comparison Summary', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.config['results_dir'], 'comprehensive_evaluation.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved evaluation plot to {output_path}")
        plt.close()


def train_with_evaluation():
    """Main training pipeline with comprehensive evaluation."""
    # Check HuggingFace token
    if not CONFIG['hf_token']:
        logger.error("HUGGINGFACE_TOKEN environment variable not set!")
        return
    
    # Setup directories
    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
    os.makedirs(CONFIG['tensorboard_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)
    
    # Initialize TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tb_writer = SummaryWriter(f"{CONFIG['tensorboard_dir']}/run_{timestamp}")
    
    # Log configuration
    tb_writer.add_text('config', json.dumps(CONFIG, indent=2))
    
    # Initialize trainer
    logger.info("Initializing stable GRPO trainer...")
    trainer = StableGRPOTrainer(
        model_name=CONFIG['base_model_name'],
        learning_rate=CONFIG['learning_rate_math'],
        batch_size=CONFIG['batch_size'],
        k_samples=CONFIG['k_samples'],
        temperature=CONFIG['temperature'],
        kl_coef=CONFIG['kl_coef'],
        max_grad_norm=CONFIG['max_grad_norm'],
        min_temperature=CONFIG['min_temperature'],
        num_epochs=1
    )
    
    # Apply GPU stability fixes
    trainer.model = apply_all_stability_fixes(trainer.model, trainer.device)
    
    # Initialize HuggingFace uploader
    hf_uploader = HuggingFaceUploader(CONFIG['hf_username'], CONFIG['hf_token'])
    
    try:
        # Phase 1: Mathematical pretraining
        logger.info("\n" + "="*70)
        logger.info("PHASE 1: MATHEMATICAL PRETRAINING")
        logger.info("="*70)
        
        # Load math data
        math_loader = MathDatasetLoader(
            dataset_name=CONFIG['math_dataset'],
            max_samples=CONFIG['math_samples']
        )
        math_data = math_loader.load_dataset()
        math_prompts = math_loader.create_prompts(math_data, include_cot=True)
        
        # Train on math (simplified for this example)
        logger.info(f"Training on {len(math_prompts)} math problems...")
        trainer.train(
            train_prompts=[p['prompt'] for p in math_prompts[:100]],  # Subset for demo
            save_path=f"{CONFIG['checkpoint_dir']}/math"
        )
        
        # Save math checkpoint
        math_checkpoint_path = f"{CONFIG['checkpoint_dir']}/math_final.pt"
        trainer.save_checkpoint(math_checkpoint_path)
        
        # Upload math model
        hf_uploader.create_and_upload_model(
            trainer.model,
            trainer.tokenizer,
            f"{CONFIG['repo_base_name']}-math",
            create_model_card(
                f"{CONFIG['repo_base_name']}-math",
                CONFIG['math_dataset'],
                {'phase': 'math'},
                CONFIG
            )
        )
        
        # Phase 2: Medical fine-tuning
        logger.info("\n" + "="*70)
        logger.info("PHASE 2: MEDICAL TRANSFER LEARNING")
        logger.info("="*70)
        
        # Load medical data
        medical_data = load_medical_datasets(
            datasets=['medmcqa', 'pubmedqa'],
            max_samples=CONFIG['medical_samples']
        )
        
        # Convert to prompts
        medical_prompts = []
        for item in medical_data:
            prompt = f"Question: {item['question']}\n\nLet me analyze this step by step.\n\n"
            medical_prompts.append(prompt)
        
        # Update learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = CONFIG['learning_rate_medical']
        
        # Train on medical (simplified)
        logger.info(f"Training on {len(medical_prompts)} medical problems...")
        trainer.train(
            train_prompts=medical_prompts[:100],  # Subset for demo
            save_path=f"{CONFIG['checkpoint_dir']}/medical"
        )
        
        # Save final checkpoint
        final_checkpoint_path = f"{CONFIG['checkpoint_dir']}/medical_final.pt"
        trainer.save_checkpoint(final_checkpoint_path)
        
        # Phase 3: Comprehensive Evaluation
        logger.info("\n" + "="*70)
        logger.info("PHASE 3: COMPREHENSIVE EVALUATION")
        logger.info("="*70)
        
        evaluator = ComprehensiveEvaluator(CONFIG)
        
        # Evaluate all models
        eval_results = evaluator.evaluate_all_models(
            base_model_path=CONFIG['base_model_name'],
            math_model_path=trainer.model,  # Current model after math training
            final_model_path=trainer.model,  # Current model after medical training
            tb_writer=tb_writer
        )
        
        # Export TensorBoard graphs
        tb_writer.close()
        export_tensorboard_to_images(
            f"{CONFIG['tensorboard_dir']}/run_{timestamp}",
            CONFIG['results_dir']
        )
        
        # Upload final model with comprehensive results
        final_metrics = {
            'accuracy': np.mean([eval_results['final'][b]['accuracy'] for b in evaluator.benchmarks]),
            'avg_tokens': np.mean([eval_results['final'][b]['avg_tokens'] for b in evaluator.benchmarks]),
            'multi_agent_quality': eval_results.get('multi_agent', {}).get('final', {}).get('consensus_quality', 0)
        }
        
        # Create enhanced model card
        enhanced_model_card = create_model_card(
            f"{CONFIG['repo_base_name']}-final",
            CONFIG['math_dataset'],
            final_metrics,
            CONFIG
        )
        
        # Add evaluation results to model card
        enhanced_model_card += "\n\n## Comprehensive Evaluation Results\n\n"
        enhanced_model_card += "### Single-Model Performance\n"
        for model in ['base', 'math', 'final']:
            if model in eval_results:
                enhanced_model_card += f"\n**{model.capitalize()} Model:**\n"
                for benchmark, results in eval_results[model].items():
                    enhanced_model_card += f"- {benchmark}: {results['accuracy']:.2%} accuracy, {results['avg_tokens']:.0f} avg tokens\n"
        
        if 'multi_agent' in eval_results:
            enhanced_model_card += "\n### Multi-Agent RL Performance\n"
            for model, results in eval_results['multi_agent'].items():
                enhanced_model_card += f"\n**{model.capitalize()} Model:**\n"
                enhanced_model_card += f"- Multi-agent accuracy: {results['accuracy']:.2%}\n"
                enhanced_model_card += f"- Token savings: {results['token_savings']:.1%}\n"
                enhanced_model_card += f"- Consensus quality: {results['consensus_quality']:.2f}\n"
        
        # Upload final model with all artifacts
        additional_files = [
            final_checkpoint_path,
            f"{CONFIG['results_dir']}/comprehensive_evaluation.png",
            f"{CONFIG['results_dir']}/math_metrics.png",
            f"{CONFIG['results_dir']}/medical_metrics.png",
            'complete_training.log'
        ]
        
        hf_uploader.create_and_upload_model(
            trainer.model,
            trainer.tokenizer,
            f"{CONFIG['repo_base_name']}-final",
            enhanced_model_card,
            additional_files=[f for f in additional_files if os.path.exists(f)]
        )
        
        # Save all results
        final_results = {
            'timestamp': timestamp,
            'config': CONFIG,
            'evaluation_results': eval_results,
            'huggingface_repos': {
                'base': CONFIG['base_model_name'],
                'math': f"{CONFIG['hf_username']}/{CONFIG['repo_base_name']}-math",
                'final': f"{CONFIG['hf_username']}/{CONFIG['repo_base_name']}-final"
            }
        }
        
        with open(f"{CONFIG['results_dir']}/complete_training_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING AND EVALUATION COMPLETE!")
        logger.info("="*70)
        logger.info("\nKey Results:")
        logger.info(f"- Base model accuracy: {np.mean([eval_results['base'][b]['accuracy'] for b in evaluator.benchmarks]):.2%}")
        logger.info(f"- Math model accuracy: {np.mean([eval_results['math'][b]['accuracy'] for b in evaluator.benchmarks]):.2%}")
        logger.info(f"- Final model accuracy: {np.mean([eval_results['final'][b]['accuracy'] for b in evaluator.benchmarks]):.2%}")
        
        if 'multi_agent' in eval_results:
            logger.info("\nMulti-Agent RL Results:")
            for model in ['base', 'math', 'final']:
                if model in eval_results['multi_agent']:
                    ma = eval_results['multi_agent'][model]
                    logger.info(f"- {model}: {ma['accuracy']:.2%} accuracy, {ma['token_savings']:.1%} token savings")
        
        logger.info(f"\nModels uploaded to HuggingFace:")
        logger.info(f"- Math: https://huggingface.co/{CONFIG['hf_username']}/{CONFIG['repo_base_name']}-math")
        logger.info(f"- Final: https://huggingface.co/{CONFIG['hf_username']}/{CONFIG['repo_base_name']}-final")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        tb_writer.close()
        raise


if __name__ == "__main__":
    train_with_evaluation()
