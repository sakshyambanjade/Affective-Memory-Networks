
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json, random
from datetime import datetime
from src.agent.agent import AMNAgent
from src.agent.baseline import BaselineAgent
from src.agent.recency import RecencyAgent
from src.agent.rag import SemanticRAGAgent

AGENTS = {
    'amn': AMNAgent(model="gpt-oss:120b-cloud"),
    'baseline': BaselineAgent(model="gpt-oss:120b-cloud"),
    'recency': RecencyAgent(model="gpt-oss:120b-cloud"),
    'semantic_rag': SemanticRAGAgent(model="gpt-oss:120b-cloud")
}

TOPICS = [
    "career_crisis", "grief_loss", "relationship_conflict", 
    "wedding_planning", "financial_stress"
] * 6  # 30 total

def generate_career_crisis_turns(n=50):
    stages = [
        "I'm feeling lost in my career right now.",
        "My boss yelled at me today, I feel worthless.",
        "I got a job offer but I'm scared to leave.",
        "I used to be excited about my projects but now everything feels hopeless.",
        "Should I just quit? What would you do in my situation?",
        "I don't know if I can handle another year here.",
        "Remember when I told you about that promotion I wanted?",
        "Now I'm second guessing everything about my career path."
    ]
    return (stages * (n // len(stages) + 1))[:n]



def run_experiment1(full=True, model="gpt-oss:120b-cloud", n_convos=100, output=None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"   # avoids warnings
    # Reduce default number of conversations to 10 for lower memory usage
    if n_convos > 10:
        print("[INFO] Reducing n_convos to 10 to avoid OOM on low-memory systems.")
        n_convos = 10
    from experiments.eval_metrics import compute_all_metrics
    # Dynamically create agents with the specified model
    AGENTS = {
        'amn': AMNAgent(model=model),
        'baseline': BaselineAgent(model=model),
        'recency': RecencyAgent(model=model),
        'semantic_rag': SemanticRAGAgent(model=model)
    }
    # Use more topics if full, else default to 30
    if full:
        topics = TOPICS * ((n_convos // len(TOPICS)) + 1)
        topics = topics[:n_convos]
    else:
        topics = TOPICS[:n_convos]
    results = []
    metrics_summary = {}
    for i, topic in enumerate(topics):
        print(f"Convo {i+1}/{len(topics)}: {topic}")
        user_turns = generate_career_crisis_turns(50)
        convo_results = {}
        convo_metrics = {}
        for condition, agent in AGENTS.items():
            print(f"  Running {condition}...")
            convo = []
            for turn, user_input in enumerate(user_turns, 1):
                if turn > 50: break
                response = agent.step(user_input)
                convo.append({
                    "turn": turn, "user": user_input, 
                    "agent": response, "condition": condition
                })
            convo_results[condition] = convo
            # Compute metrics for this condition
            convo_metrics[condition] = compute_all_metrics(convo)
        results.append({
            "convo_id": i+1,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            **convo_results
        })
        metrics_summary[f"convo_{i+1}"] = convo_metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    if output is None:
        filename = f'results/exp1_30convos_{timestamp}.json'
    else:
        filename = output
    metrics_file = f'results/exp1_metrics_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=1)
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=1)
    print(f"✅ Experiment 1 COMPLETE: {filename}")
    print(f"Total turns: {len(results)*50*4} across 4 conditions")
    print(f"Metrics saved: {metrics_file}")
    return filename

if __name__ == "__main__":
    run_experiment1()
