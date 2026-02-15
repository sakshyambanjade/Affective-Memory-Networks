import json
HUMAN_METRICS = {
    'empathy': 'How empathetic was the agent response? (1-5)',
    'coherence': 'How coherent with conversation history? (1-5)',
    'trust': 'How trustworthy did the agent seem? (1-5)'
}

def generate_eval_pairs(convo_file):
    with open(convo_file) as f:
        data = json.load(f)
    eval_pairs = []
    for convo in data:
        for condition in ['amn', 'baseline']:
            turns = convo[condition]
            critical_turns = [turns[i] for i in [9,19,29,39,49]]
            for turn in critical_turns:
                eval_pairs.append({
                    'convo_id': convo['convo_id'],
                    'condition': condition,
                    'turn': turn['turn'],
                    'history': ' | '.join([f"T{t}: {x['user']}" for t,x in enumerate(turns[:turn['turn']])]),
                    'response': turn['agent'],
                    'metrics': HUMAN_METRICS
                })
    with open('results/human_eval_pairs.json', 'w') as f:
        json.dump(eval_pairs[:100], f)
    print(f"✅ Human eval ready: 100 pairs for n=20 evaluators")
