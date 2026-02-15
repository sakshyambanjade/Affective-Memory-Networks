import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results/prolific_eval_results.csv')

pivot_empathy = df.pivot(index='pair_id', columns='condition', values='empathy')
pivot_coherence = df.pivot(index='pair_id', columns='condition', values='coherence')
pivot_trust = df.pivot(index='pair_id', columns='condition', values='trust')

print("HUMAN EVALUATION RESULTS (n=20 evaluators)")
print("\nEmpathy (1-5 Likert):")
print(pivot_empathy.mean().round(2))

print("\nRepeated Measures ANOVA:")
for metric in ['empathy', 'coherence', 'trust']:
    pivot = df.pivot(index='pair_id', columns='condition', values=metric)
    f_stat, p_val = stats.f_oneway(*[pivot[col].dropna() for col in pivot.columns])
    print(f"{metric}: F={f_stat:.1f}, p={p_val:.4f} {'***' if p_val<0.05 else ''}")

def cohens_d(group1, group2):
    return (group1.mean() - group2.mean()) / np.sqrt((group1.var() + group2.var()) / 2)

amn_emp = df[df.condition=='amn']['empathy']
base_emp = df[df.condition=='baseline']['empathy']
print(f"\nEffect Sizes (AMN vs Baseline):")
print(f"Empathy d={cohens_d(amn_emp, base_emp):.2f}")
