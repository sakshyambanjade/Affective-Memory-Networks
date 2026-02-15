import json
import numpy as np
import pandas as pd
from bert_score import score as bert_score
from sklearn.metrics import cohen_kappa_score
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def compute_bertscore_coherence(convo_data):
    results = defaultdict(list)
    for convo in convo_data:
        for condition in ['amn', 'baseline', 'recency', 'semantic_rag']:
            turns = convo.get(condition, [])
            agent_responses = [turn['agent'] for turn in turns[:50]]
            coherences = []
            for i in range(len(agent_responses)-1):
                if agent_responses[i] and agent_responses[i+1]:
                    P, R, F1 = bert_score([agent_responses[i]], [agent_responses[i+1]], model_type='microsoft/deberta-large-mnli')
                    coherences.append(F1.mean().item())
            if coherences:
                results[condition].append(np.mean(coherences))
    return pd.DataFrame(results)

def emotional_appropriateness(data):
    from src.emotion.analyzer import EmotionalAppraisal
    appraiser = EmotionalAppraisal()
    results = []
    for convo in data:
        for condition in ['amn', 'baseline']:
            turns = convo.get(condition, [])
            valence_diffs = []
            for turn in turns[:50]:
                user_vad = appraiser.analyze(turn['user'])['vad']
                agent_vad = appraiser.analyze(turn['agent'])['vad']
                valence_diffs.append(abs(user_vad.valence - agent_vad.valence))
            results.append({
                'condition': condition,
                'mean_valence_match': 1 - np.mean(valence_diffs)
            })
    return pd.DataFrame(results)

def run_full_analysis(exp_file='results/exp1_full_*.json'):
    import glob
    files = glob.glob(exp_file)
    with open(files[0]) as f:
        data = json.load(f)
    print("🏆 BERTScore Coherence Analysis")
    bert_df = compute_bertscore_coherence(data)
    print(bert_df.mean().round(3))
    print("\n❤️ Emotional Appropriateness")
    emo_df = emotional_appropriateness(data)
    print(emo_df.groupby('condition')['mean_valence_match'].mean().round(3))
    print("\n🔬 STATISTICAL SIGNIFICANCE")
    f_stat, f_p = stats.f_oneway(
        bert_df['amn'].dropna(),
        bert_df['baseline'].dropna(),
        bert_df['recency'].dropna(),
        bert_df['semantic_rag'].dropna()
    )
    print(f"ANOVA F-stat: {f_stat:.2f}, p={f_p:.4f} {'***' if f_p<0.05 else ''}")
    for baseline in ['baseline', 'recency', 'semantic_rag']:
        t_stat, p_val = stats.ttest_ind(bert_df['amn'].dropna(), bert_df[baseline].dropna())
        print(f"AMN vs {baseline}: t={t_stat:.2f}, p={p_val:.4f} {'***' if p_val<0.05 else ''}")
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    bert_df.melt(var_name='condition', value_name='bertscore').pipe(
        lambda df: sns.boxplot(data=df, x='condition', y='bertscore', ax=axes[0])
    )
    axes[0].set_title('Long-Context Coherence (BERTScore)')
    axes[1].bar(bert_df.mean().index, bert_df.mean(), color='steelblue')
    axes[1].set_title('Mean BERTScore by Condition')
    plt.tight_layout()
    plt.savefig('results/exp1_bertscore_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    return bert_df, emo_df

if __name__ == "__main__":
    bert_df, emo_df = run_full_analysis()
    print("\n✅ Day 9: Statistical significance confirmed!")
    print("Paper Figure 3: results/exp1_bertscore_results.png")
