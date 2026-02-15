import json, re
import numpy as np
from collections import Counter

def memory_reference_rate(convo_file):
    memory_cues = ['remember', 'past', 'before', 'earlier', 'used to', 'last time']
    with open(convo_file) as f:
        data = json.load(f)
    results = {'amn': [], 'baseline': [], 'recency': [], 'semantic_rag': []}
    for convo in data:
        for condition in results.keys():
            turns = convo.get(condition, [])
            refs = 0
            total = 0
            for turn in turns:
                resp = turn['agent'].lower()
                if any(cue in resp for cue in memory_cues):
                    refs += 1
                total += 1
            results[condition].append(refs / total if total else 0)
    means = {k: np.mean(v) for k,v in results.items()}
    print("Memory Reference Rate:")
    for cond, rate in means.items():
        print(f"  {cond}: {rate:.1%}")
    return means

# Run on Day 4/5 data
memory_reference_rate('results/exp1_30convos_*.json')
