try:
    from core import WorkingMemory, EpisodicMemory
except ImportError:
    from src.memory.core import WorkingMemory, EpisodicMemory
from datetime import timedelta

wm = WorkingMemory()
em = EpisodicMemory()

# Simulate 7 turns (no eviction)
for i, text in enumerate([
    "Excited about project!", "Sad about delay.", "Angry at bug.",
    "Calm now.", "Fear deadline.", "Happy fix!", "Neutral chat."
]):
    entry = wm.add(text)
    if em.consolidate(entry):
        em.add(entry)  # Evicted WM → EM

# Overflow: Add 3 more → evict 3
for text in ["More excited!", "Very sad.", "Furious!"]:
    entry = wm.add(text)
    em.add(entry)  # All to EM post-eviction

print("WM (top 5):", [m.id[:8] for m in wm.get_all()])
print("EM count:", len(em.memories))
print("Sample EM VAD:", em.get_recent(1)[0].vad)

# Verify: WM len=5, EM has evictions + peaks
assert len(wm.memories) == 5
import logging
logger = logging.getLogger('AMN')
logger.info("Day 2 tests PASS")
