#!/usr/bin/env python3
"""Create comparison plots for GRPO evaluation results"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

# Load the results
with open('results/grpo_evaluation_results.json', 'r') as f:
    data = json.load(f)

# Prepare data for plotting
models = []
baseline_acc = []
baseline_tokens = []
baseline_eff = []
grpo_acc = []
grpo_tokens = []
grpo_eff = []

for result in data['grpo_results']:
    model_name = result['model_name']
    models.append(model_name.replace('Qwen/Qwen2.5-', ''))
    
    # Get baseline data
    baseline = data['baseline_results'][model_name]
    baseline_acc.append(baseline['accuracy'])
    baseline_tokens.append(baseline['tokens'])
    baseline_eff.append(baseline['efficiency'])
    
    # Get GRPO data
    grpo_acc.append(result['overall_accuracy'])
    grpo_tokens.append(result['overall_avg_tokens'])
    grpo_eff.append(result['efficiency_score'])

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GRPO Training Impact on Medical Reasoning Performance', fontsize=16)

# 1. Accuracy Comparison
ax1 = axes[0, 0]
x = np.arange(len(models))
width = 0.35
bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline', color='#1f77b4')
bars2 = ax1.bar(x + width/2, grpo_acc, width, label='GRPO-Trained', color='#ff7f0e')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom')

ax1.set_ylabel('Accuracy')
ax1.set_title('Model Accuracy Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. Token Efficiency
ax2 = axes[0, 1]
bars1 = ax2.bar(x - width/2, baseline_tokens, width, label='Baseline', color='#1f77b4')
bars2 = ax2.bar(x + width/2, grpo_tokens, width, label='GRPO-Trained', color='#ff7f0e')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

ax2.set_ylabel('Average Tokens per Response')
ax2.set_title('Token Usage Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Efficiency Score
ax3 = axes[1, 0]
bars1 = ax3.bar(x - width/2, baseline_eff, width, label='Baseline', color='#1f77b4')
bars2 = ax3.bar(x + width/2, grpo_eff, width, label='GRPO-Trained', color='#ff7f0e')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')

ax3.set_ylabel('Efficiency Score (Accuracy / Token Usage)')
ax3.set_title('Overall Efficiency Comparison')
ax3.set_xticks(x)
ax3.set_xticklabels(models, rotation=45)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Improvement Summary
ax4 = axes[1, 1]
improvements = data['improvements']
improvement_data = []
for model in ['Qwen/Qwen2.5-0.5B-Instruct', 'Qwen/Qwen2.5-1.5B-Instruct', 
              'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B']:
    imp = improvements[model]
    improvement_data.append({
        'Model': model.replace('Qwen/Qwen2.5-', ''),
        'Accuracy Gain': imp['accuracy_gain'] * 100,
        'Token Reduction': imp['token_reduction'],
        'Efficiency Gain': imp['efficiency_gain']
    })

# Create grouped bar chart for improvements
imp_df = pd.DataFrame(improvement_data)
imp_x = np.arange(len(imp_df))
width2 = 0.25

bars1 = ax4.bar(imp_x - width2, imp_df['Accuracy Gain'], width2, label='Accuracy Gain (%)', color='#2ca02c')
bars2 = ax4.bar(imp_x, imp_df['Token Reduction'], width2, label='Token Reduction', color='#d62728')
bars3 = ax4.bar(imp_x + width2, imp_df['Efficiency Gain'] * 10, width2, label='Efficiency Gain (×10)', color='#9467bd')

ax4.set_ylabel('Improvement Metrics')
ax4.set_title('GRPO Training Improvements')
ax4.set_xticks(imp_x)
ax4.set_xticklabels(imp_df['Model'], rotation=45)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/grpo_comparison_plot.png', dpi=300, bbox_inches='tight')
print("GRPO comparison plot saved to results/grpo_comparison_plot.png")

# Create a simplified summary plot for the report
fig2, ax = plt.subplots(figsize=(10, 6))

# Create grouped bar chart
x = np.arange(len(models))
width = 0.2

# Plot metrics
bars1 = ax.bar(x - 1.5*width, [b*100 for b in baseline_acc], width, label='Baseline Accuracy', color='#1f77b4', alpha=0.7)
bars2 = ax.bar(x - 0.5*width, [g*100 for g in grpo_acc], width, label='GRPO Accuracy', color='#1f77b4')
bars3 = ax.bar(x + 0.5*width, baseline_tokens, width, label='Baseline Tokens', color='#ff7f0e', alpha=0.7)
bars4 = ax.bar(x + 1.5*width, grpo_tokens, width, label='GRPO Tokens', color='#ff7f0e')

# Add second y-axis for tokens
ax2 = ax.twinx()
ax2.set_ylabel('Average Tokens per Response', color='#ff7f0e')
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

ax.set_xlabel('Model Variant')
ax.set_ylabel('Accuracy (%)', color='#1f77b4')
ax.tick_params(axis='y', labelcolor='#1f77b4')
ax.set_title('GRPO Training Results: Accuracy vs Token Efficiency')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45)
ax.grid(axis='y', alpha=0.3)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.7, label='Baseline Accuracy'),
    Patch(facecolor='#1f77b4', label='GRPO Accuracy'),
    Patch(facecolor='#ff7f0e', alpha=0.7, label='Baseline Tokens'),
    Patch(facecolor='#ff7f0e', label='GRPO Tokens')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('results/grpo_summary_plot.png', dpi=300, bbox_inches='tight')
print("GRPO summary plot saved to results/grpo_summary_plot.png")
