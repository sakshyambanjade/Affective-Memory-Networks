import json, numpy as np, pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_experiment_data(pattern='results/exp1_*.json'):
    import glob
    files = glob.glob(pattern)
    all_data = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
            all_data.extend(data)
    return all_data

def memory_reference_analysis(data):
    memory_cues = ['remember', 'past', 'before', 'earlier', 'used to', 'last time']
    results = []
    for convo in data:
        for condition in ['amn', 'baseline', 'recency', 'semantic_rag']:
            turns = convo.get(condition, [])
            refs, total = 0, 0
            for turn in turns[:50]:
                if any(cue in turn['agent'].lower() for cue in memory_cues):
                    refs += 1
                total += 1
            results.append({
                'condition': condition,
                'convo_id': convo['convo_id'],
                'ref_rate': refs/total if total else 0
            })
    df = pd.DataFrame(results)
    print("\n🎯 MEMORY REFERENCE RATE (Key Phase 1 Metric)")
    print(df.groupby('condition')['ref_rate'].agg(['mean', 'std', 'count']))
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x='condition', y='ref_rate')
    plt.title('Memory Reference Rate by Condition')
    plt.ylabel('Proportion of responses referencing past')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/memory_ref_rate.png', dpi=300)
    plt.show()
    return df

def emotional_continuity_analysis(data):
    from src.emotion.analyzer import EmotionalAppraisal
    appraiser = EmotionalAppraisal()
    results = []
    for convo in data:
        for condition in ['amn', 'baseline']:
            turns = convo.get(condition, [])
            user_vads = []
            for turn in turns[:50]:
                vad_dict = appraiser.analyze(turn['user'])['vad']
                user_vads.append([vad_dict.valence, vad_dict.arousal])
            if len(user_vads) > 10:
                valence_std = np.std([v[0] for v in user_vads])
                results.append({
                    'condition': condition,
                    'valence_consistency': 1-valence_std
                })
    df = pd.DataFrame(results)
    print("\n📊 EMOTIONAL ARC CONSISTENCY")
    print(df.groupby('condition')['valence_consistency'].mean())
    return df

if __name__ == "__main__":
    data = load_experiment_data()
    memory_df = memory_reference_analysis(data)
    emo_df = emotional_continuity_analysis(data)
    amn_refs = memory_df[memory_df.condition=='amn']['ref_rate']
    base_refs = memory_df[memory_df.condition=='baseline']['ref_rate']
    t_stat, p_val = stats.ttest_ind(amn_refs, base_refs)
    print(f"\n🔬 T-TEST: AMN vs Baseline Memory Refs")
    print(f"p-value: {p_val:.4f} {'***' if p_val<0.05 else ''}")
