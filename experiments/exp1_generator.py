import json, random
from datetime import datetime
from src.agent.agent import AMNAgent
from src.agent.baseline import BaselineAgent
from src.agent.recency import RecencyAgent
from src.agent.rag import SemanticRAGAgent

AGENTS = {
    'amn': AMNAgent(),
    'baseline': BaselineAgent(),
    'recency': RecencyAgent(),
    'semantic_rag': SemanticRAGAgent()
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

def run_experiment1():
    results = []
    for i, topic in enumerate(TOPICS):
        print(f"Convo {i+1}/30: {topic}")
        user_turns = generate_career_crisis_turns(50)
        convo_results = {}
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
        results.append({
            "convo_id": i+1,
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            **convo_results
        })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f'results/exp1_30convos_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=1)
    print(f"✅ Experiment 1 COMPLETE: {filename}")
    print(f"Total turns: {len(results)*50*4} across 4 conditions")
    return filename

if __name__ == "__main__":
    run_experiment1()
