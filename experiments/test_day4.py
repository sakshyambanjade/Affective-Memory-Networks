from src.agent.agent import AMNAgent
from src.agent.baseline import BaselineAgent
import json
from datetime import datetime

career_crisis = [
    "I'm feeling lost in my career. Not sure what to do next.",
    "My boss yelled at me today. I feel worthless.",
    "I got a job offer but scared to leave.",
    "Remember that time I was excited about my project?",
    "Now everything feels hopeless again.",
    "Should I quit? What would you do?",
] * 2  # 50+ turns

def run_conversation(agent, topic: str, turns: int = 50):
    convo = []
    user_turns = career_crisis[:turns//2] if topic == "career" else ["How are you?"] * turns
    for i, user_input in enumerate(user_turns):
        response = agent.step(user_input)
        convo.append({"turn": i+1, "user": user_input, "agent": response})
        if i >= turns: break
    return convo

amn_agent = AMNAgent()
baseline_agent = BaselineAgent()

print("Running AMN...")
amn_convo = run_conversation(amn_agent, "career", 30)
print("Running Baseline...")
baseline_convo = run_conversation(baseline_agent, "career", 30)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
with open(f'results/convo_day4_{timestamp}.json', 'w') as f:
    json.dump({"amn": amn_convo, "baseline": baseline_convo}, f, indent=2)

print(f"✅ Day 4: 30-turn convos saved to results/")
print(f"AMN memories: WM={len(amn_agent.wm.memories)}, EM={len(amn_agent.em.memories)}")
