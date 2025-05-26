#!/usr/bin/env python3
"""Create comprehensive plots for milestone report"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read the baseline results
df = pd.read_csv('results/baseline_results.csv')

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# 1. Accuracy by Model Type and Size
ax1 = plt.subplot(2, 3, 1)
model_data = []
for _, row in df.groupby(['model_type', 'model'])[['accuracy']].mean().iterrows():
    model_type, model_name = _
    size = '0.5B' if '0.5B' in model_name else '1.5B'
    type_label = 'Base' if 'base' in model_type else 'Instruct'
    model_data.append({
        'Model Size': size,
        'Model Type': type_label,
        'Accuracy': row['accuracy']
    })

plot_df = pd.DataFrame(model_data)
pivot_acc = plot_df.pivot(index='Model Size', columns='Model Type', values='Accuracy')
pivot_acc.plot(kind='bar', ax=ax1, color=['#FF6B6B', '#4ECDC4'])
ax1.set_title('Accuracy by Model Size and Type', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.legend(title='Model Type')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

# 2. Token Efficiency
ax2 = plt.subplot(2, 3, 2)
token_data = []
for _, row in df.groupby(['model_type', 'model'])[['avg_tokens']].mean().iterrows():
    model_type, model_name = _
    size = '0.5B' if '0.5B' in model_name else '1.5B'
    type_label = 'Base' if 'base' in model_type else 'Instruct'
    token_data.append({
        'Model Size': size,
        'Model Type': type_label,
        'Avg Tokens': row['avg_tokens']
    })

token_df = pd.DataFrame(token_data)
pivot_tokens = token_df.pivot(index='Model Size', columns='Model Type', values='Avg Tokens')
pivot_tokens.plot(kind='bar', ax=ax2, color=['#FF6B6B', '#4ECDC4'])
ax2.set_title('Average Token Usage', fontsize=14, fontweight='bold')
ax2.set_ylabel('Tokens')
ax2.legend(title='Model Type')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)

# 3. Accuracy by Dataset
ax3 = plt.subplot(2, 3, 3)
dataset_acc = df.pivot_table(index='dataset', columns='model_type', values='accuracy')
dataset_acc.plot(kind='bar', ax=ax3, color=['#FFD93D', '#6BCF7F', '#4ECDC4', '#FF6B6B'])
ax3.set_title('Accuracy by Dataset', fontsize=14, fontweight='bold')
ax3.set_ylabel('Accuracy')
ax3.set_xlabel('Dataset')
ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Model Type')
ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

# 4. Efficiency Score (Accuracy per Token)
ax4 = plt.subplot(2, 3, 4)
efficiency_data = []
for idx, row in df.iterrows():
    model_name = row['model'].split('/')[-1]
    efficiency_score = row['accuracy'] / (row['avg_tokens'] / 100)
    efficiency_data.append({
        'Model': model_name,
        'Efficiency Score': efficiency_score,
        'Type': 'Base' if 'base' in row['model_type'] else 'Instruct'
    })

eff_df = pd.DataFrame(efficiency_data)
eff_df_grouped = eff_df.groupby('Model')['Efficiency Score'].mean().sort_values(ascending=False)
colors = ['#FF6B6B' if 'Instruct' not in m else '#4ECDC4' for m in eff_df_grouped.index]
eff_df_grouped.plot(kind='bar', ax=ax4, color=colors)
ax4.set_title('Efficiency Score (Accuracy/Tokens×100)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Score')
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

# 5. Scatter: Accuracy vs Tokens
ax5 = plt.subplot(2, 3, 5)
for model_type in df['model_type'].unique():
    mask = df['model_type'] == model_type
    color = '#FF6B6B' if 'base' in model_type else '#4ECDC4'
    marker = 'o' if '0.5B' in model_type else 's'
    ax5.scatter(df[mask]['avg_tokens'], df[mask]['accuracy'], 
               label=model_type, alpha=0.7, s=150, c=color, marker=marker, edgecolors='black')

ax5.set_xlabel('Average Tokens')
ax5.set_ylabel('Accuracy')
ax5.set_title('Token Efficiency vs Accuracy Trade-off', fontsize=14, fontweight='bold')
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)

# Add efficiency frontier
base_points = df[df['model_type'].str.contains('base')][['avg_tokens', 'accuracy']].values
instruct_points = df[df['model_type'].str.contains('instruct')][['avg_tokens', 'accuracy']].values
ax5.plot([base_points[:, 0].mean(), instruct_points[:, 0].mean()], 
         [base_points[:, 1].mean(), instruct_points[:, 1].mean()], 
         'k--', alpha=0.5, label='Trade-off line')

# 6. Summary Statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary table
summary_data = []
for model_type in ['base-0.5B', 'base-1.5B', 'instruct-0.5B', 'instruct-1.5B']:
    type_data = df[df['model_type'] == model_type]
    if not type_data.empty:
        summary_data.append({
            'Model': model_type,
            'Accuracy': f"{type_data['accuracy'].mean():.1%}",
            'Tokens': f"{type_data['avg_tokens'].mean():.1f}",
            'Efficiency': f"{type_data['accuracy'].mean() / (type_data['avg_tokens'].mean() / 100):.2f}"
        })

summary_df = pd.DataFrame(summary_data)
table = ax6.table(cellText=summary_df.values,
                  colLabels=summary_df.columns,
                  cellLoc='center',
                  loc='center',
                  bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 2)

# Style the table
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4ECDC4')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        if 'base' in summary_df.iloc[i-1]['Model']:
            table[(i, j)].set_facecolor('#FFE5E5')
        else:
            table[(i, j)].set_facecolor('#E5F7F7')

ax6.set_title('Summary Statistics (500 examples per dataset)', fontsize=14, fontweight='bold', pad=20)

plt.suptitle('MedLogicTrace Full Experiment Results - 500 Examples per Dataset', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/milestone_full_results.png', dpi=300, bbox_inches='tight')
print("Saved milestone_full_results.png")

# Create a simpler comparison plot for the report
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
model_types = ['Base\n0.5B', 'Base\n1.5B', 'Instruct\n0.5B', 'Instruct\n1.5B']
accuracies = [0.635, 0.598, 0.843, 0.819]
tokens = [29.9, 21.5, 64.0, 64.0]

colors = ['#FF6B6B', '#FF6B6B', '#4ECDC4', '#4ECDC4']
bars1 = ax1.bar(model_types, accuracies, color=colors, alpha=0.8, edgecolor='black')
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 1)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')

# Token usage comparison
bars2 = ax2.bar(model_types, tokens, color=colors, alpha=0.8, edgecolor='black')
ax2.set_ylabel('Average Tokens', fontsize=12)
ax2.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 80)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, tok in zip(bars2, tokens):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{tok:.1f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('MedLogicTrace: Accuracy vs Token Efficiency Trade-off\n(500 examples each from MedMCQA and PubMedQA)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/milestone_comparison.png', dpi=300, bbox_inches='tight')
print("Saved milestone_comparison.png")

plt.close('all')
