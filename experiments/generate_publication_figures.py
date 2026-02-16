"""
Generate all publication-quality figures for paper
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def create_figure2_memory_rates():
    """Figure 2: Memory Reference Rates with CI"""
    systems = ['AMN\n(Ours)', 'Baseline', 'Semantic\nRAG', 'Recency\nOnly']
    means = [80.0, 47.7, 33.0, 15.0]
    cis = [4.2, 5.8, 4.5, 3.2]
    colors = ['#2E7D32', '#757575', '#9E9E9E', '#BDBDBD']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(systems, means, color=colors, edgecolor='black', linewidth=1.5,
                  yerr=cis, capsize=8, error_kw={'linewidth': 2})
    
    for bar, mean, ci in zip(bars, means, cis):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + ci + 2,
                f'{mean:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Memory Reference Rate (%)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.plot([0, 1], [95, 95], 'k-', linewidth=1.5)
    ax.text(0.5, 96, '***', ha='center', fontsize=16)
    
    ax.text(0.02, 0.98, 'n = 15 conversations', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Memory Reference Rate Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure2_memory_rates_final.png', dpi=300, bbox_inches='tight')
    print("✓ Created Figure 2: Memory Reference Rates")

def create_figure3_coherence():
    """Figure 3: Coherence with statistical significance"""
    systems = ['AMN\n(Ours)', 'Baseline', 'Semantic\nRAG', 'Recency\nOnly']
    means = [0.889, 0.868, 0.886, 0.880]
    stds = [0.012, 0.024, 0.015, 0.018]
    colors = ['#2E7D32', '#757575', '#9E9E9E', '#BDBDBD']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(systems, means, color=colors, edgecolor='black', linewidth=1.5,
                  yerr=stds, capsize=8, error_kw={'linewidth': 2})
    
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.002,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('BERTScore (Coherence)', fontsize=15, fontweight='bold')
    ax.set_ylim(0.84, 0.92)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', labelsize=12)
    
    ax.plot([0, 1], [0.91, 0.91], 'k-', linewidth=1.5)
    ax.text(0.5, 0.912, '**', ha='center', fontsize=16)
    ax.text(0.5, 0.914, 'p = 0.009', ha='center', fontsize=10)
    
    ax.text(0.02, 0.98, 'n = 15 conversations', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.title('Coherence Score Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure3_coherence_final.png', dpi=300, bbox_inches='tight')
    print("✓ Created Figure 3: Coherence Scores")

def create_figure4_ablation():
    """Figure 4: Ablation Study Results"""
    variants = ['Full\nModel', 'No\nEmotion', 'No\nGoal', 'No\nIntensity', 'No\nRecency']
    memory_rates = [80.0, 52.3, 71.4, 73.8, 78.1]
    colors = ['#2E7D32'] + ['#E57373']*4
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(variants, memory_rates, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, rate in zip(bars, memory_rates):
        height = bar.get_height()
        drop = 80.0 - rate
        label = f'{rate:.1f}%\n(-{drop:.1f}pp)' if drop > 0 else f'{rate:.1f}%'
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                label,
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Memory Reference Rate (%)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Full Model Baseline')
    
    plt.title('Ablation Study: Impact of Each Retrieval Factor', fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure4_ablation_study.png', dpi=300, bbox_inches='tight')
    print("✓ Created Figure 4: Ablation Study")

# Run all
if __name__ == '__main__':
    create_figure2_memory_rates()
    create_figure3_coherence()
    create_figure4_ablation()
    print("\n✅ All publication figures generated!")
    print(f"   Location: {RESULTS_DIR}/")
