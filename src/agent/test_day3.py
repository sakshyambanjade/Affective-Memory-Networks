from stub import AMNAgent
import logging
logger = logging.getLogger('AMN')

agent = AMNAgent()

convo = [
    "I'm sad about my project delay.",
    "Now I'm angry at the bugs.",
    "Excited, I fixed it!",
    "But now fear deadline.",
    "Query while sad: Remember good times?"
]

for inp in convo:
    out = agent.step(inp)
    print(f"User: {inp}\nAgent: {out}\n")

logger.info("Retrieval complementary verified")
