# AMN Test Cases

## Test Case 1: VAD Analyzer Accuracy

### Test Sentences with Expected VAD Scores

```json
[
  {
    "text": "I'm so happy today!",
    "expected_vad": {
      "valence": 0.85,
      "arousal": 0.70,
      "dominance": 0.70
    },
    "tolerance": 0.15
  },
  {
    "text": "I feel terrible and alone.",
    "expected_vad": {
      "valence": -0.75,
      "arousal": 0.45,
      "dominance": 0.35
    },
    "tolerance": 0.15
  },
  {
    "text": "I'm terrified about the interview tomorrow.",
    "expected_vad": {
      "valence": -0.65,
      "arousal": 0.85,
      "dominance": 0.30
    },
    "tolerance": 0.15
  },
  {
    "text": "That makes me so angry!",
    "expected_vad": {
      "valence": -0.70,
      "arousal": 0.90,
      "dominance": 0.70
    },
    "tolerance": 0.15
  },
  {
    "text": "I'm feeling calm and content.",
    "expected_vad": {
      "valence": 0.75,
      "arousal": 0.25,
      "dominance": 0.65
    },
    "tolerance": 0.15
  }
]
```

## Test Case 2: Memory Storage and Retrieval

### Test Scenario: Emotional Memory Priority

```python
# User has a conversation about job loss (negative, high arousal)
# Then conversation about new opportunities (positive, medium arousal)
# Then asks about career - should retrieve both with emotional context

test_conversation = [
    {
        "turn": 1,
        "user": "I just lost my job and I'm devastated.",
        "expected_memory": True,
        "expected_importance": 0.9  # High arousal + high negative valence
    },
    {
        "turn": 2,
        "user": "I'm worried about my financial situation.",
        "expected_memory": True,
        "expected_importance": 0.7
    },
    {
        "turn": 3,
        "user": "But I've been applying to some exciting new positions.",
        "expected_memory": True,
        "expected_importance": 0.8  # Positive shift
    },
    {
        "turn": 4,
        "user": "What should I focus on for my career?",
        "expected_retrieval": [1, 2, 3],  # Should retrieve all career-related memories
        "emotional_context": "Should acknowledge both loss and new opportunities"
    }
]
```

## Test Case 3: Emotional Resonance

### Test: Complementary Emotional Retrieval

```python
# When user is sad, system should retrieve positive memories for regulation
test_emotional_resonance = {
    "current_state": {
        "valence": -0.7,  # User is sad
        "arousal": 0.5,
        "dominance": 0.3
    },
    "memory_pool": [
        {
            "id": "mem_001",
            "text": "I got the promotion!",
            "valence": 0.9,
            "arousal": 0.7,
            "dominance": 0.8
        },
        {
            "id": "mem_002",
            "text": "I failed the exam.",
            "valence": -0.6,
            "arousal": 0.4,
            "dominance": 0.3
        },
        {
            "id": "mem_003",
            "text": "I had a great time with friends.",
            "valence": 0.8,
            "arousal": 0.6,
            "dominance": 0.7
        }
    ],
    "expected_top_retrieval": ["mem_001", "mem_003"],  # Positive memories for regulation
    "mode": "complementary"
}
```

## Test Case 4: Consolidation Trigger

### Test: High-arousal memories should trigger consolidation

```python
test_consolidation = [
    {
        "memory": "I lost my grandmother.",
        "valence": -0.85,
        "arousal": 0.75,  # High arousal
        "goal_relevance": 0.65,
        "should_consolidate": True  # arousal > 0.7
    },
    {
        "memory": "I had lunch today.",
        "valence": 0.2,
        "arousal": 0.15,  # Low arousal
        "goal_relevance": 0.1,
        "should_consolidate": False  # Both thresholds not met
    },
    {
        "memory": "I finally finished my thesis!",
        "valence": 0.88,
        "arousal": 0.68,
        "goal_relevance": 0.95,  # High goal relevance
        "should_consolidate": True  # goal_relevance > 0.8
    }
]
```

## Test Case 5: Multi-turn Coherence

### Test: Agent should reference past emotional context

```python
test_coherence = {
    "conversation": [
        {"turn": 1, "user": "My dog died yesterday.", "emotion": "grief"},
        {"turn": 2, "user": "I can't focus on anything.", "emotion": "distracted"},
        {"turn": 3, "user": "Should I get a new pet?", "emotion": "uncertain"}
    ],
    "expected_agent_behavior": {
        "turn": 3,
        "should_reference_turn_1": True,
        "appropriate_response_markers": [
            "I know you're grieving",
            "losing your dog",
            "when you're ready"
        ],
        "inappropriate_response_markers": [
            "that's great!",
            "just get a new one",
            "move on"
        ]
    }
}
```

## Test Case 6: Baseline Comparison

### Test: AMN vs. Baseline on same conversation

```python
test_baseline_comparison = {
    "conversation": [
        "I'm so anxious about my presentation tomorrow.",
        "What if I forget everything?",
        "I've practiced but I'm still terrified.",
        "Do you have any advice?"
    ],
    "expected_differences": {
        "amn": {
            "should_reference": "your anxiety",
            "emotional_awareness": "high",
            "coherence_score": "> 0.8"
        },
        "baseline": {
            "should_reference": "the presentation",
            "emotional_awareness": "minimal",
            "coherence_score": "< 0.6"
        }
    }
}
```

## Validation Criteria

### Phase 1 Success Criteria

- [ ] VAD analyzer accuracy > 80% within tolerance
- [ ] Memory storage saves all interactions with emotion tags
- [ ] Retrieval prioritizes high-importance memories
- [ ] Emotional resonance mode (complementary) works
- [ ] Consolidation triggers for arousal > 0.7 OR goal_relevance > 0.8
- [ ] AMN shows statistical improvement over baseline (p < 0.05)
- [ ] Human evaluators rate AMN empathy > 3.5/5.0

### Expected Performance Metrics

| Metric | Baseline | AMN Target | Method |
|--------|----------|------------|---------|
| BERTScore Coherence | 0.60-0.65 | > 0.80 | Automated |
| Memory Reference Rate | < 10% | > 40% | Count explicit references |
| Emotional Appropriateness | 0.50-0.55 | > 0.75 | VAD matching |
| Human Empathy Rating | 2.5-3.0 | > 4.0 | Likert scale 1-5 |
| Human Coherence Rating | 2.8-3.2 | > 4.0 | Likert scale 1-5 |

## Running Tests

```bash
# Test 1: VAD Analyzer
python test_cases/test_vad_accuracy.py

# Test 2: Memory System
python test_cases/test_memory_system.py

# Test 3: Emotional Retrieval
python test_cases/test_emotional_retrieval.py

# Test 4: Full Integration
python test_cases/test_full_system.py

# Test 5: Run all tests
python test_cases/run_all_tests.py
```

## Debugging Checklist

If tests fail:

1. **VAD scores are off**
   - Check lexicon loaded correctly
   - Verify word coverage (should have 2000+ words)
   - Test with known-emotion sentences

2. **Memories not stored**
   - Check Working Memory capacity (should be 5)
   - Verify eviction to Episodic Memory works
   - Check timestamp and emotion tags present

3. **Retrieval returns wrong memories**
   - Verify scoring weights sum to 1.0
   - Check emotional resonance mode (complementary vs similar)
   - Test semantic similarity component

4. **Agent responses lack coherence**
   - Check retrieved memories are being used in prompt
   - Verify context formatting
   - Test LLM API connection

5. **Consolidation not triggered**
   - Check thresholds: arousal > 0.7 OR goal_relevance > 0.8
   - Verify appraisal scores computed correctly
   - Check consolidation interval (not just threshold)
