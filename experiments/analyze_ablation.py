"""
Analyze ablation study results
Computes memory reference rates and coherence for each variant
"""

import json
import numpy as np
from pathlib import Path
from bert_score import score as bert_score
import re

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / 'results' / 'ablation'

def count_memory_references(response):
    """Check if response references past conversation"""
    if not response:
        return False
    
    reference_patterns = [
        r'\b(remember|recall|mentioned|said|told|earlier|before|previously)\b',
        r'\b(last time|when you)\b',
        r'\b(you (?:said|mentioned|told))\b'
    ]
    
    for pattern in reference_patterns:
        if re.search(pattern, response.lower()):
            return True
    return False

def compute_coherence(conversation_turns):
    """Compute BERTScore coherence for a conversation"""
    if not conversation_turns:
        return 0.0
    
    references = [turn['user'] for turn in conversation_turns if 'user' in turn]
    candidates = [turn['agent'] for turn in conversation_turns 
                  if 'agent' in turn and turn['agent']]
    
    if not candidates or not references:
        return 0.0
    
    references_expanded = references * (len(candidates) // len(references) + 1)
    references_expanded = references_expanded[:len(candidates)]
    
    try:
        P, R, F1 = bert_score(candidates, references_expanded, lang='en', verbose=False)
        return F1.mean().item()
    except:
        return 0.0

def analyze_variant(variant_name, variant_results):
    """Analyze one ablation variant"""
    print(f"\n{'='*50}")
    print(f"Analyzing: {variant_name}")
    print(f"{'='*50}")
    
    memory_refs = []
    coherence_scores = []
    
    for convo in variant_results:
        turns = convo['turns']
        refs = sum(1 for turn in turns if count_memory_references(turn.get('agent')))
        total_turns = len([t for t in turns if 'agent' in t and t['agent']])
        if total_turns > 0:
            ref_rate = refs / total_turns
            memory_refs.append(ref_rate)
        coh = compute_coherence(turns)
        if coh > 0:
            coherence_scores.append(coh)
    
    memory_mean = np.mean(memory_refs) * 100 if memory_refs else 0
    memory_std = np.std(memory_refs) * 100 if memory_refs else 0
    coherence_mean = np.mean(coherence_scores) if coherence_scores else 0
    coherence_std = np.std(coherence_scores) if coherence_scores else 0
    
    print(f"Memory Reference Rate: {memory_mean:.1f}% ± {memory_std:.1f}%")
    print(f"Coherence (BERTScore): {coherence_mean:.3f} ± {coherence_std:.3f}")
    
    return {
        'variant': variant_name,
        'memory_rate_mean': memory_mean,
        'memory_rate_std': memory_std,
        'coherence_mean': coherence_mean,
        'coherence_std': coherence_std,
        'n_conversations': len(variant_results)
    }

def main():
    result_files = sorted(RESULTS_DIR.glob('ablation_results_*.json'))
    if not result_files:
        print("❌ No ablation results found. Run ablation_study.py first.")
        return
    
    latest_file = result_files[-1]
    print(f"📊 Analyzing: {latest_file}")
    
    with open(latest_file) as f:
        data = json.load(f)
    
    results = data['results']
    configs = data['configs']
    
    summary = []
    for variant_name, variant_results in results.items():
        stats = analyze_variant(variant_name, variant_results)
        summary.append(stats)
    
    print(f"\n{'='*70}")
    print(f"ABLATION STUDY SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<20} {'Memory Rate':<20} {'Coherence':<20}")
    print(f"{'-'*70}")
    summary.sort(key=lambda x: x['memory_rate_mean'], reverse=True)
    for stats in summary:
        print(f"{stats['variant']:<20} "
              f"{stats['memory_rate_mean']:>5.1f}% ± {stats['memory_rate_std']:>4.1f}%    "
              f"{stats['coherence_mean']:>5.3f} ± {stats['coherence_std']:>5.3f}")
    full_stats = next(s for s in summary if s['variant'] == 'full')
    print(f"\n{'='*70}")
    print(f"PERFORMANCE DROPS (vs Full Model)")
    print(f"{'='*70}")
    for stats in summary:
        if stats['variant'] == 'full':
            continue
        drop = full_stats['memory_rate_mean'] - stats['memory_rate_mean']
        print(f"{stats['variant']:<20} Δ = {drop:>+6.1f}pp")
    summary_file = RESULTS_DIR / f'ablation_summary_{latest_file.stem.split("_")[-1]}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'summary': summary,
            'full_model': full_stats,
            'configs': configs
        }, f, indent=2)
    print(f"\n✅ Analysis complete: {summary_file}")
    print(f"\n📊 Use this data for Table 2 in paper:")
    print(f"   'Removing emotional relevance drops performance by {full_stats['memory_rate_mean'] - next(s for s in summary if s['variant']=='no_emotional')['memory_rate_mean']:.1f}pp'")

if __name__ == '__main__':
    main()
