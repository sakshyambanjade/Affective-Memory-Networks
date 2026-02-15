from src.agent.agent import AMNAgent

agent = AMNAgent()

test_texts = [
    "I achieved my biggest career goal today!",
    "The deadline is tomorrow and I haven't started.",
    "My boss unexpectedly praised my work."
]

for text in test_texts:
    result = agent.appraiser.full_appraisal(text)
    print(f"Text: {text}")
    print(f"Goal: {result['lazarus'].goal_relevance:.2f}, Consolidate: {result['consolidate']}")
    print()
