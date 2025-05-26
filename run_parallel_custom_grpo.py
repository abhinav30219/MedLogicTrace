#!/usr/bin/env python3
"""Parallel GRPO training across multiple GPUs with live progress tracking"""

import os
import sys
import subprocess
import threading
import time
from datetime import datetime
import json
from pathlib import Path

class ParallelGRPOTrainer:
    def __init__(self):
        self.progress_file = "grpo_progress.txt"
        # Use all 8 GPUs - 2 per model for data parallelism
        self.models_and_gpus = [
            ("Qwen/Qwen2.5-0.5B-Instruct", 0, "split_0"),
            ("Qwen/Qwen2.5-0.5B-Instruct", 1, "split_1"),
            ("Qwen/Qwen2.5-1.5B-Instruct", 2, "split_0"),
            ("Qwen/Qwen2.5-1.5B-Instruct", 3, "split_1"),
            ("Qwen/Qwen2.5-0.5B", 4, "split_0"),
            ("Qwen/Qwen2.5-0.5B", 5, "split_1"),
            ("Qwen/Qwen2.5-1.5B", 6, "split_0"),
            ("Qwen/Qwen2.5-1.5B", 7, "split_1"),
        ]
        self.processes = {}
        self.progress_data = {}
        self.start_time = None
        
    def update_progress_display(self):
        """Update the progress file with current status"""
        with open(self.progress_file, 'w') as f:
            f.write(f"GRPO Parallel Training Progress - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            total_models = len(self.models_and_gpus)
            completed = sum(1 for data in self.progress_data.values() if data.get('status') == 'Completed')
            
            f.write(f"Overall Progress: {completed}/{total_models} models completed\n")
            if self.start_time:
                elapsed = time.time() - self.start_time
                f.write(f"Total Time Elapsed: {int(elapsed//60)}m {int(elapsed%60)}s\n")
            f.write("\n" + "-" * 80 + "\n\n")
            
            for model_info in self.models_and_gpus:
                model_name, gpu_id, split = model_info
                model_key = f"{model_name}_gpu{gpu_id}"
                progress = self.progress_data.get(model_key, {})
                
                f.write(f"Model: {model_name} (GPU {gpu_id}, {split})\n")
                f.write(f"Status: {progress.get('status', 'Waiting')}\n")
                
                if progress.get('epoch'):
                    f.write(f"Progress: Epoch {progress['epoch']}/{progress.get('total_epochs', '?')} | ")
                    f.write(f"Batch {progress.get('batch', '?')}/{progress.get('total_batches', '?')}\n")
                
                if progress.get('current_reward') is not None:
                    f.write(f"Current Reward: {progress['current_reward']:.3f} | ")
                    f.write(f"Best: {progress.get('best_reward', 0.0):.3f}\n")
                
                if progress.get('elapsed_time'):
                    f.write(f"Time: {progress['elapsed_time']} | ETA: {progress.get('eta', 'calculating...')}\n")
                
                if progress.get('last_example'):
                    f.write(f"Last Output Sample: {progress['last_example'][:100]}...\n")
                    
                f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("Monitor live with: watch -n 2 'tail -n 50 grpo_progress.txt'\n")
    
    def parse_training_output(self, line, model_key):
        """Parse training output and update progress"""
        line = line.strip()
        
        # Parse different types of output
        if "Epoch" in line and "/" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "Epoch" and i+1 < len(parts):
                    epoch_info = parts[i+1].split('/')
                    if len(epoch_info) == 2:
                        self.progress_data[model_key]['epoch'] = epoch_info[0]
                        self.progress_data[model_key]['total_epochs'] = epoch_info[1]
        
        elif "Batch" in line and "Avg Reward:" in line:
            # Extract batch info and reward
            import re
            batch_match = re.search(r'Batch (\d+)/(\d+)', line)
            reward_match = re.search(r'Avg Reward: ([\d.]+)', line)
            
            if batch_match:
                self.progress_data[model_key]['batch'] = batch_match.group(1)
                self.progress_data[model_key]['total_batches'] = batch_match.group(2)
            
            if reward_match:
                current_reward = float(reward_match.group(1))
                self.progress_data[model_key]['current_reward'] = current_reward
                best = self.progress_data[model_key].get('best_reward', 0.0)
                self.progress_data[model_key]['best_reward'] = max(best, current_reward)
        
        elif "Response:" in line:
            # Capture example output
            self.progress_data[model_key]['last_example'] = line.split("Response:", 1)[1].strip()
        
        elif "Training completed" in line:
            self.progress_data[model_key]['status'] = 'Completed'
        
        elif "Initial evaluation" in line:
            self.progress_data[model_key]['status'] = 'Evaluating'
        
        elif "Starting GRPO training" in line:
            self.progress_data[model_key]['status'] = 'Training'
    
    def monitor_process(self, process, model_name, gpu_id):
        """Monitor a single training process"""
        model_key = f"{model_name}_gpu{gpu_id}"
        self.progress_data[model_key] = {
            'status': 'Starting',
            'start_time': time.time()
        }
        
        # Create individual log file
        log_file = f"logs/grpo_{model_name.replace('/', '_')}_gpu{gpu_id}.log"
        os.makedirs("logs", exist_ok=True)
        
        with open(log_file, 'w') as log:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                    
                # Write to individual log
                log.write(line)
                log.flush()
                
                # Parse and update progress
                self.parse_training_output(line, model_key)
                
                # Update elapsed time
                elapsed = time.time() - self.progress_data[model_key]['start_time']
                self.progress_data[model_key]['elapsed_time'] = f"{int(elapsed//60)}m {int(elapsed%60)}s"
                
                # Update progress display
                self.update_progress_display()
        
        # Mark as completed
        self.progress_data[model_key]['status'] = 'Completed'
        self.update_progress_display()
    
    def run_parallel_training(self):
        """Launch and monitor all training processes"""
        print("="*80)
        print("Parallel GRPO Training with Live Progress Tracking")
        print(f"Starting {len(self.models_and_gpus)} models across GPUs")
        print("="*80)
        
        self.start_time = time.time()
        threads = []
        
        # Launch all processes
        for model_info in self.models_and_gpus:
            model_name, gpu_id, split = model_info
            print(f"Launching {model_name} on GPU {gpu_id} ({split})...")
            
            # Create the training command
            cmd = [
                sys.executable,
                "run_single_grpo_training.py",
                "--model_name", model_name,
                "--gpu_id", str(gpu_id),
                "--num_epochs", "1",
                "--k_responses", "4",
                "--data_split", split
            ]
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
            )
            
            self.processes[f"{model_name}_gpu{gpu_id}"] = process
            
            # Create monitoring thread
            thread = threading.Thread(
                target=self.monitor_process,
                args=(process, model_name, gpu_id),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            
            # Small delay to avoid race conditions
            time.sleep(2)
        
        print(f"\nAll models launched! Monitor progress in: {self.progress_file}")
        print(f"Or run: watch -n 2 'tail -n 50 {self.progress_file}'")
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Final summary
        total_time = time.time() - self.start_time
        print(f"\n{'='*80}")
        print(f"All training completed in {int(total_time//60)}m {int(total_time%60)}s")
        
        # Save final summary
        summary = {
            'total_time_minutes': total_time / 60,
            'models_trained': len(self.models_and_gpus),
            'final_results': self.progress_data
        }
        
        with open('grpo_parallel_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: grpo_parallel_summary.json")

def main():
    # First, create the single model training script
    create_single_training_script()
    
    # Then run parallel training
    trainer = ParallelGRPOTrainer()
    trainer.run_parallel_training()

def create_single_training_script():
    """Create a script for training a single model"""
    script_content = '''#!/usr/bin/env python3
"""Single model GRPO training script (called by parallel trainer)"""

import sys
sys.path.append('src')
import argparse
from run_custom_grpo_training import train_grpo_model, create_reasoning_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--k_responses', type=int, default=4)
    parser.add_argument('--data_split', type=str, default=None)
    args = parser.parse_args()
    
    # If data_split is specified, only use half the data
    if args.data_split:
        # Create full dataset and split it
        full_dataset = create_reasoning_dataset(100)
        
        # Split data in half
        if args.data_split == "split_0":
            train_data = full_dataset[:50]
        else:  # split_1
            train_data = full_dataset[50:]
        
        print(f"Using {args.data_split} with {len(train_data)} samples")
    else:
        train_data = None  # Use default
    
    # Run training with custom data if provided
    output_dir, summary = train_grpo_model(
        model_name=args.model_name,
        gpu_id=args.gpu_id,
        num_epochs=args.num_epochs,
        K=args.k_responses,
        custom_train_data=train_data
    )
    
    print(f"\\nTraining completed for {args.model_name} on GPU {args.gpu_id}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
'''
    
    with open('run_single_grpo_training.py', 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod('run_single_grpo_training.py', 0o755)

if __name__ == "__main__":
    main()
