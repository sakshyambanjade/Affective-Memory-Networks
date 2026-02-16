"""
Fixed Memory Reference Counter
Checks if agent response references ANY past conversation content
"""

import json
import re
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).parent.parent

def extract_key_phrases(text, min_length=4):
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'from', 'by', 'about', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them', 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how'}
    words = text.lower().split()
    phrases = []
    current_phrase = []
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word and clean_word not in stopwords:
            current_phrase.append(clean_word)
        elif current_phrase:
            if len(current_phrase) >= min_length:
                phrases.append(' '.join(current_phrase))
            current_phrase = []
    if current_phrase and len(current_phrase) >= min_length:
        phrases.append(' '.join(current_phrase))
    return phrases

def similarity_ratio(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

def check_memory_reference_improved(agent_response, past_user_inputs):
    if not agent_response or not past_user_inputs:
        return False
    explicit_patterns = [
        r'\b(remember|recall|mentioned|said|told|earlier|before|previously)\b',
        r'\b(last time|when you)\b',
        r'\b(you (?:said|mentioned|told|talked about))\b',
        r'\b(as (?:we|you) (?:discussed|talked about))\b'
    ]
    for pattern in explicit_patterns:
        if re.search(pattern, agent_response.lower()):
            return True
    response_phrases = extract_key_phrases(agent_response)
    for past_input in past_user_inputs:
        past_phrases = extract_key_phrases(past_input)
        for past_phrase in past_phrases:
            for response_phrase in response_phrases:
                if similarity_ratio(past_phrase, response_phrase) > 0.7:
                    return True
    for past_input in past_user_inputs:
        past_words = set(word.lower() for word in re.findall(r'\b[A-Z][a-z]+\b', past_input))
        past_words.update(re.findall(r'\b(?:job|work|relationship|family|friend|doctor|hospital|school|project)\b', past_input.lower()))
        for word in past_words:
            if word in agent_response.lower():
                return True
    return False

def recount_all_results():
    print("🔍 RECOUNTING MEMORY REFERENCES WITH IMPROVED METHOD")
    print("="*60)
    results_file = PROJECT_ROOT / 'results' / 'exp1_realdata_20260216_0135.json'
    with open(results_file) as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} conversations\n")
    system_counts = {'amn': [], 'baseline': [], 'semantic_rag': [], 'recency': []}
    for i, convo in enumerate(data):
        print(f"[{i+1}/{len(data)}] {convo['primary_emotion']}")
        for system in ['amn', 'baseline', 'semantic_rag', 'recency']:
            if system in convo and convo[system]:
                turns = convo[system]
                past_inputs = []
                refs_count = 0
                total_turns = 0
                for turn in turns:
                    if 'user' in turn and 'agent' in turn:
                        user_text = turn['user']
                        agent_text = turn.get('agent', '')
                        if agent_text:
                            if check_memory_reference_improved(agent_text, past_inputs):
                                refs_count += 1
                            total_turns += 1
                        past_inputs.append(user_text)
                if total_turns > 0:
                    ref_rate = (refs_count / total_turns) * 100
                    system_counts[system].append(ref_rate)
                    print(f"  {system}: {refs_count}/{total_turns} = {ref_rate:.1f}%")
    print(f"\n{'='*60}")
    print(f"RECOUNTED RESULTS")
    print(f"{'='*60}")
    for system in ['amn', 'baseline', 'semantic_rag', 'recency']:
        if system_counts[system]:
            mean = sum(system_counts[system]) / len(system_counts[system])
            print(f"{system:15s}: {mean:>5.1f}%")
    output = {
        'method': 'improved_memory_counting',
        'per_conversation': system_counts,
        'averages': {
            system: sum(counts)/len(counts) if counts else 0
            for system, counts in system_counts.items()
        }
    }
    output_file = PROJECT_ROOT / 'results' / 'recounted_memory_refs.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✅ Saved to: {output_file}")

if __name__ == '__main__':
    recount_all_results()
