"""
Per-Emotion Analysis: Which emotions benefit most from AMN?
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results'

# Load results
with open(RESULTS_DIR / 'exp1_realdata_20260216_0135.json') as f:
    data = json.load(f)

print(f"✓ Loaded {len(data)} conversations")

# Group by emotion
emotion_groups = defaultdict(lambda: {'amn': [], 'baseline': [], 'semantic_rag': [], 'recency': []})

for convo in data:
    emotion = convo['primary_emotion']
    # Count memory references for each system
    for system in ['amn', 'baseline', 'semantic_rag', 'recency']:
        if system in convo and convo[system]:
            turns = convo[system]
            refs = sum(1 for turn in turns if turn.get('references_memory', False))
            total = len(turns)
            ref_rate = refs / total if total > 0 else 0
            emotion_groups[emotion][system].append(ref_rate * 100)

# Compute statistics per emotion
emotion_stats = []
for emotion, systems in emotion_groups.items():
    amn_mean = np.mean(systems['amn']) if systems['amn'] else 0
    baseline_mean = np.mean(systems['baseline']) if systems['baseline'] else 0
    improvement = amn_mean - baseline_mean
    emotion_stats.append({
        'emotion': emotion,
        'amn_mean': amn_mean,
        'baseline_mean': baseline_mean,
        'improvement': improvement,
        'n': len(systems['amn'])
    })

# Sort by improvement
emotion_stats.sort(key=lambda x: x['improvement'], reverse=True)

# Print table
print(f"\n{'='*70}")
print(f"PER-EMOTION ANALYSIS")
print(f"{'='*70}")
print(f"{'Emotion':<15} {'AMN':<10} {'Baseline':<10} {'Improvement':<15} {'N':<5}")
print(f"{'-'*70}")
for stat in emotion_stats:
    print(f"{stat['emotion']:<15} "
          f"{stat['amn_mean']:>6.1f}%   "
          f"{stat['baseline_mean']:>6.1f}%   "
          f"{stat['improvement']:>+6.1f}pp       "
          f"{stat['n']:>3}")

# Categorize emotions by valence/arousal
emotion_categories = {
    'negative_high': ['grief', 'anxiety', 'anger', 'fear', 'embarrassment'],
    'negative_low': ['disappointment', 'loneliness', 'guilt', 'sadness'],
    'positive_high': ['joy', 'excitement', 'pride'],
    'positive_low': ['gratitude', 'contentment', 'relief', 'hope']
}

# Group stats by category
category_stats = {}
for category, emotions in emotion_categories.items():
    cat_data = [s for s in emotion_stats if s['emotion'] in emotions]
    if cat_data:
        category_stats[category] = {
            'amn': np.mean([s['amn_mean'] for s in cat_data]),
            'baseline': np.mean([s['baseline_mean'] for s in cat_data]),
            'improvement': np.mean([s['improvement'] for s in cat_data])
        }

print(f"\n{'='*70}")
print(f"BY CATEGORY (Valence-Arousal)")
print(f"{'='*70}")
for category, stats in category_stats.items():
    print(f"{category:<20} AMN: {stats['amn']:.1f}%  "
          f"Baseline: {stats['baseline']:.1f}%  "
          f"Δ = {stats['improvement']:+.1f}pp")

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Per-emotion breakdown
emotions = [s['emotion'] for s in emotion_stats]
amn_vals = [s['amn_mean'] for s in emotion_stats]
baseline_vals = [s['baseline_mean'] for s in emotion_stats]

x = np.arange(len(emotions))
width = 0.35

ax1.bar(x - width/2, amn_vals, width, label='AMN', color='#2E7D32')
ax1.bar(x + width/2, baseline_vals, width, label='Baseline', color='#757575')

ax1.set_ylabel('Memory Reference Rate (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Emotion', fontsize=12, fontweight='bold')
ax1.set_title('Memory Reference Rate by Emotion', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(emotions, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 100)

# Plot 2: Category comparison
categories = list(category_stats.keys())
cat_amn = [category_stats[c]['amn'] for c in categories]
cat_baseline = [category_stats[c]['baseline'] for c in categories]
cat_labels = ['Neg/High', 'Neg/Low', 'Pos/High', 'Pos/Low']

x2 = np.arange(len(categories))
ax2.bar(x2 - width/2, cat_amn, width, label='AMN', color='#2E7D32')
ax2.bar(x2 + width/2, cat_baseline, width, label='Baseline', color='#757575')

ax2.set_ylabel('Memory Reference Rate (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Emotion Category', fontsize=12, fontweight='bold')
ax2.set_title('Performance by Emotion Category', fontsize=14, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(cat_labels)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 100)

plt.tight_layout()
output_file = RESULTS_DIR / 'figure4_per_emotion_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Figure saved: {output_file}")

# Save statistics
with open(RESULTS_DIR / 'per_emotion_stats.json', 'w') as f:
    json.dump({
        'per_emotion': emotion_stats,
        'by_category': category_stats
    }, f, indent=2)

print(f"\n📊 Key Finding for Paper:")
best_category = max(category_stats.items(), key=lambda x: x[1]['improvement'])
print(f"   '{best_category[0]} emotions show largest improvement: "
      f"+{best_category[1]['improvement']:.1f}pp'")
