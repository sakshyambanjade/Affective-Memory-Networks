import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t
import matplotlib.pyplot as plt
import seaborn as sns

# Compile all metrics into master results table
results = pd.DataFrame({
    'condition': ['amn']*30 + ['baseline']*30 + ['recency']*30 + ['semantic_rag']*30,
    'bertscore': np.concatenate([amn_bert, base_bert, recency_bert, rag_bert]),
    'memory_refs': np.concatenate([amn_refs, base_refs, recency_refs, rag_refs]),
    'empathy': np.concatenate([amn_emp, base_emp, recency_emp, rag_emp])
})

def mean_ci(data):
    m = data.mean()
    sem = data.sem()
    ci = sem * t.ppf((1 + 0.95) / 2., len(data)-1)
    return f"{m:.3f} ({m-ci:.3f}, {m+ci:.3f})"

summary = results.groupby('condition').apply(
    lambda g: pd.Series({
        'BERTScore': mean_ci(g.bertscore),
        'Memory Refs': f"{g.memory_refs.mean():.1%}",
        'Empathy': mean_ci(g.empathy)
    })
)
print("TABLE 1: Final Results with 95% CIs")
print(summary.round(3))
summary.to_latex('results/final_results_table.tex')
