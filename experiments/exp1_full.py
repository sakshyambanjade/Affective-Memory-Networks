import multiprocessing as mp
import json
from datetime import datetime
from experiments.exp1_generator import AGENTS, TOPICS, generate_career_crisis_turns

def worker_convo(args):
    convo_id, topic = args
    user_turns = generate_career_crisis_turns(50)
    convo_results = {}
    for condition, agent in AGENTS.items():
        convo = []
        for turn, user_input in enumerate(user_turns[:50]):
            response = agent.step(user_input)
            convo.append({"turn": turn+1, "user": user_input, "agent": response})
        convo_results[condition] = convo
    return {"convo_id": convo_id, "topic": topic, **convo_results}

if __name__ == "__main__":
    with mp.Pool(4) as pool:
        results = pool.map(worker_convo, enumerate(TOPICS))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    with open(f'results/exp1_full_{timestamp}.json', 'w') as f:
        json.dump(results, f)
    print("✅ FULL Experiment 1: 30 convos complete")
