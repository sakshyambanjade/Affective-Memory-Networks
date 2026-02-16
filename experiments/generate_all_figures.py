import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Load results
results_files = glob.glob('results/exp1_realdata_*.json')
if not results_files:
    print("No results files found! Run experiments first.")
    exit(1)

with open(results_files[0], 'r') as f:
    data = json.load(f)

print(f"Loaded {len(data)} conversations")

# ============================================================
# FIGURE 1: BERTSCORE COHERENCE COMPARISON
# ============================================================

try:
    from bert_score import score as bert_score
except ImportError:
    print("bert-score package not found. Please install it with 'pip install bert-score'.")
    exit(1)

def compute_bertscore(conversations):
    results = {'amn': [], 'baseline': [], 'recency': [], 'semantic_rag': []}
    for conv in conversations:
        for condition in results.keys():
            if condition not in conv:
                continue
            turns = conv[condition]
            responses = [t.get('agent', '') for t in turns if t.get('agent')]
            if len(responses) < 2:
                continue
            coherences = []
            for i in range(len(responses) - 1):
                if responses[i] and responses[i+1]:
                    P, R, F1 = bert_score([responses[i]], [responses[i+1]], lang='en', verbose=False)
                    coherences.append(F1.item())
            if coherences:
                results[condition].append(np.mean(coherences))
    return results

print("Computing BERTScore coherence...")
coherence_scores = compute_bertscore(data)

# Plot Figure 1
fig, ax = plt.subplots(figsize=(10, 6))
conditions = list(coherence_scores.keys())
means = [np.mean(coherence_scores[c]) for c in conditions]
stds = [np.std(coherence_scores[c]) for c in conditions]
colors = ['#2ecc71', '#3498db', '#f39c12', '#95a5a6']
bars = ax.bar(conditions, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('BERTScore F1 (Coherence)', fontsize=14, fontweight='bold')
ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
ax.set_title('Long-Context Coherence: AMN vs Baselines', fontsize=16, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
for i, (bar, mean) in enumerate(zip(bars, means)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('results/figure1_coherence.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure1_coherence.png")
plt.close()

# ============================================================
# FIGURE 2: MEMORY REFERENCE RATE
# ============================================================

def compute_memory_references(conversations):
    results = {'amn': [], 'baseline': [], 'recency': [], 'semantic_rag': []}
    memory_keywords = [
        'remember', 'recalled', 'earlier', 'before', 'previously',
        'you said', 'you mentioned', 'last time', 'you told me',
        'we discussed', 'as you said', 'you felt'
    ]
    for conv in conversations:
        for condition in results.keys():
            if condition not in conv:
                continue
            turns = conv[condition]
            total_responses = 0
            memory_refs = 0
            for turn in turns:
                if turn is None or not isinstance(turn, dict):
                    continue
                agent_response = turn.get('agent', '')
                if isinstance(agent_response, str):
                    agent_response_lower = agent_response.lower()
                else:
                    agent_response_lower = ''
                if agent_response_lower:
                    total_responses += 1
                    if any(keyword in agent_response_lower for keyword in memory_keywords):
                        memory_refs += 1
            if total_responses > 0:
                ref_rate = (memory_refs / total_responses) * 100
                results[condition].append(ref_rate)
    return results

print("Computing memory reference rates...")
memory_refs = compute_memory_references(data)

# Plot Figure 2
fig, ax = plt.subplots(figsize=(10, 6))
conditions = list(memory_refs.keys())
means = [np.mean(memory_refs[c]) if memory_refs[c] else 0 for c in conditions]
stds = [np.std(memory_refs[c]) if memory_refs[c] else 0 for c in conditions]
bars = ax.bar(conditions, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Memory Reference Rate (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Condition', fontsize=14, fontweight='bold')
ax.set_title('Explicit Memory References in Agent Responses', fontsize=16, fontweight='bold')
ax.set_ylim([0, max(means) * 1.3])
ax.grid(axis='y', alpha=0.3)
for bar, mean in zip(bars, means):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('results/figure2_memory_refs.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure2_memory_refs.png")
plt.close()

# ============================================================
# FIGURE 4: COMBINED METRICS COMPARISON
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Left: Coherence
ax = axes[0]
means_coherence = [np.mean(coherence_scores[c]) for c in conditions]
ax.bar(conditions, means_coherence, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('BERTScore F1', fontsize=12, fontweight='bold')
ax.set_title('(a) Coherence', fontsize=14, fontweight='bold')
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)
# Right: Memory References
ax = axes[1]
means_memory = [np.mean(memory_refs[c]) if memory_refs[c] else 0 for c in conditions]
ax.bar(conditions, means_memory, color=colors, alpha=0.8, edgecolor='black')
ax.set_ylabel('Reference Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('(b) Memory References', fontsize=14, fontweight='bold')
ax.set_ylim([0, max(means_memory) * 1.3])
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('results/figure4_combined_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure4_combined_metrics.png")
plt.close()

# ============================================================
# FIGURE 5: EMOTION DISTRIBUTION
# ============================================================

emotions = [conv.get('primary_emotion', 'unknown') for conv in data]
emotion_counts = pd.Series(emotions).value_counts()
fig, ax = plt.subplots(figsize=(10, 10))
colors_pie = plt.cm.Set3(range(len(emotion_counts)))
wedges, texts, autotexts = ax.pie(emotion_counts, labels=emotion_counts.index, 
                                    autopct='%1.1f%%', colors=colors_pie,
                                    startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(10)
ax.set_title('Emotion Distribution in Test Dataset', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results/figure5_emotion_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: results/figure5_emotion_distribution.png")
plt.close()

# ============================================================
# STATISTICS SUMMARY
# ============================================================

from scipy import stats
print("\n" + "="*60)
print("STATISTICAL ANALYSIS")
print("="*60)
# ANOVA
conditions_data = [coherence_scores[c] for c in conditions]
f_stat, p_value = stats.f_oneway(*conditions_data)
print(f"\nANOVA for Coherence:")
print(f"   F-statistic: {f_stat:.3f}")
print(f"   p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
# Pairwise t-tests
print(f"\nPairwise Comparisons (AMN vs others):")
for condition in ['baseline', 'recency', 'semantic_rag']:
    if condition in coherence_scores:
        t_stat, p_val = stats.ttest_ind(coherence_scores['amn'], coherence_scores[condition])
        cohens_d = (np.mean(coherence_scores['amn']) - np.mean(coherence_scores[condition])) / \
                   np.sqrt((np.std(coherence_scores['amn'])**2 + np.std(coherence_scores[condition])**2) / 2)
        print(f"   AMN vs {condition:15s}: t={t_stat:6.3f}, p={p_val:.6f} {'***' if p_val < 0.001 else ''}, d={cohens_d:.3f}")
# Print means and stds
print(f"\nCoherence Scores (Mean ± SD):")
for condition in conditions:
    mean = np.mean(coherence_scores[condition])
    std = np.std(coherence_scores[condition])
    print(f"   {condition:15s}: {mean:.3f} ± {std:.3f}")
print(f"\nMemory Reference Rates (Mean ± SD):")
for condition in conditions:
    if memory_refs[condition]:
        mean = np.mean(memory_refs[condition])
        std = np.std(memory_refs[condition])
        print(f"   {condition:15s}: {mean:.1f}% ± {std:.1f}%")
# Save statistics to file
with open('results/statistics_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("AMN EXPERIMENTAL RESULTS SUMMARY\n")
    f.write("="*60 + "\n\n")
    f.write("COHERENCE (BERTScore F1):\n")
    for condition in conditions:
        mean = np.mean(coherence_scores[condition])
        std = np.std(coherence_scores[condition])
        f.write(f"  {condition:15s}: {mean:.3f} ± {std:.3f}\n")
    f.write(f"\nSTATISTICAL SIGNIFICANCE:\n")
    f.write(f"  ANOVA: F={f_stat:.3f}, p={p_value:.6f}\n")
    f.write(f"\nMEMORY REFERENCES:\n")
    for condition in conditions:
        if memory_refs[condition]:
            mean = np.mean(memory_refs[condition])
            f.write(f"  {condition:15s}: {mean:.1f}%\n")
print("\nSaved: results/statistics_summary.txt")
print("\n" + "="*60)
print("ALL FIGURES GENERATED!")
print("="*60)
print("\nGenerated files:")
print("  results/figure1_coherence.png")
print("  results/figure2_memory_refs.png")
print("  results/figure4_combined_metrics.png")
print("  results/figure5_emotion_distribution.png")
print("  results/statistics_summary.txt")