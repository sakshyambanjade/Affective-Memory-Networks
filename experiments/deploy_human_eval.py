import json
import pandas as pd

def generate_prolific_csv(eval_pairs_file='results/human_eval_pairs.json'):
    with open(eval_pairs_file) as f:
        pairs = json.load(f)
    df = pd.DataFrame(pairs[:100])
    df['pair_id'] = range(100)
    df.to_csv('results/prolific_eval_batch1.csv', index=False)
    print("✅ Prolific CSV ready: 100 pairs")
    print("Deploy: https://app.prolific.com/studies → Upload CSV")
    print("Payment: $1.50 × 20 eval = $30 budget")

generate_prolific_csv()
