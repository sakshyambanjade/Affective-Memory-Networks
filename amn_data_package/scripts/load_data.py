#!/usr/bin/env python3
"""
AMN Data Package Loader
Easy integration of conversations and lexicon into your project
"""

import json
import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple

class AMNDataLoader:
    """Load and manage AMN data package"""
    
    def __init__(self, data_package_dir: str = "data_package"):
        self.data_dir = Path(data_package_dir)
        self.conversations_dir = self.data_dir / "conversations"
        self.lexicons_dir = self.data_dir / "lexicons"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data package not found at {data_package_dir}. "
                f"Please extract the AMN data package first."
            )
    
    def load_conversations(self, filename: str = "empathetic_dialogues_100.json") -> List[Dict]:
        """
        Load pre-processed emotional conversations.
        
        Returns:
            List of conversation dicts with structure:
            {
                'id': 'conv_001',
                'primary_emotion': 'grief',
                'turns': [
                    {'speaker': 'user', 'text': '...', 'emotion': '...', 'valence': -0.8, ...},
                    {'speaker': 'agent', 'text': '...', 'response_type': 'empathetic'}
                ]
            }
        """
        conv_path = self.conversations_dir / filename
        
        if not conv_path.exists():
            raise FileNotFoundError(f"Conversation file not found: {conv_path}")
        
        with open(conv_path, 'r') as f:
            conversations = json.load(f)
        
        print(f"✅ Loaded {len(conversations)} conversations")
        print(f"📊 Emotion distribution:")
        
        emotion_counts = {}
        for conv in conversations:
            emotion = conv['primary_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
        
        return conversations
    
    def load_vad_lexicon(self, filename: str = "warriner_vad_2000.csv") -> Dict[str, Tuple[float, float, float]]:
        """
        Load VAD lexicon: word -> (valence, arousal, dominance)
        
        Returns:
            Dict mapping words to (valence, arousal, dominance) tuples
            - Valence: -1 (negative) to +1 (positive)
            - Arousal: 0 (calm) to 1 (excited)
            - Dominance: 0 (submissive) to 1 (dominant)
        """
        lexicon_path = self.lexicons_dir / filename
        
        if not lexicon_path.exists():
            raise FileNotFoundError(f"Lexicon file not found: {lexicon_path}")
        
        lexicon = {}
        with open(lexicon_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row['Word'].lower()
                valence = float(row['Valence'])
                arousal = float(row['Arousal'])
                dominance = float(row['Dominance'])
                lexicon[word] = (valence, arousal, dominance)
        
        print(f"✅ Loaded {len(lexicon)} words in VAD lexicon")
        return lexicon
    
    def get_conversations_by_emotion(self, emotion: str) -> List[Dict]:
        """
        Filter conversations by primary emotion.
        
        Args:
            emotion: One of [grief, joy, anxiety, anger, gratitude, etc.]
        
        Returns:
            List of conversations with that primary emotion
        """
        all_convos = self.load_conversations()
        filtered = [c for c in all_convos if c['primary_emotion'] == emotion]
        print(f"Found {len(filtered)} conversations with emotion '{emotion}'")
        return filtered
    
    def get_high_arousal_conversations(self, threshold: float = 0.7) -> List[Dict]:
        """Get conversations with high emotional arousal."""
        all_convos = self.load_conversations()
        high_arousal = []
        
        for conv in all_convos:
            # Check if any turn has arousal > threshold
            for turn in conv['turns']:
                if turn['speaker'] == 'user' and turn.get('arousal', 0) > threshold:
                    high_arousal.append(conv)
                    break
        
        print(f"Found {len(high_arousal)} high-arousal conversations (arousal > {threshold})")
        return high_arousal
    
    def split_train_test(self, test_size: float = 0.2, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
        """
        Split conversations into train and test sets.
        
        Args:
            test_size: Fraction for test set (default 0.2 = 20%)
            seed: Random seed for reproducibility
        
        Returns:
            (train_conversations, test_conversations)
        """
        import random
        random.seed(seed)
        
        all_convos = self.load_conversations()
        random.shuffle(all_convos)
        
        split_idx = int(len(all_convos) * (1 - test_size))
        train = all_convos[:split_idx]
        test = all_convos[split_idx:]
        
        print(f"📂 Split: {len(train)} train, {len(test)} test")
        return train, test
    
    def prepare_for_experiment(self, n_conversations: int = 30) -> List[Dict]:
        """
        Select N diverse conversations for experiments.
        Ensures balanced emotion distribution.
        
        Args:
            n_conversations: Number of conversations to select (default 30)
        
        Returns:
            List of selected conversations
        """
        import random
        random.seed(42)
        
        all_convos = self.load_conversations()
        
        # Group by emotion
        by_emotion = {}
        for conv in all_convos:
            emotion = conv['primary_emotion']
            if emotion not in by_emotion:
                by_emotion[emotion] = []
            by_emotion[emotion].append(conv)
        
        # Sample evenly from each emotion
        selected = []
        emotions = list(by_emotion.keys())
        per_emotion = n_conversations // len(emotions)
        
        for emotion in emotions:
            available = by_emotion[emotion]
            to_select = min(per_emotion, len(available))
            selected.extend(random.sample(available, to_select))
        
        # Fill remaining with random selection
        while len(selected) < n_conversations and len(selected) < len(all_convos):
            remaining = [c for c in all_convos if c not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        print(f"✅ Selected {len(selected)} diverse conversations for experiment")
        
        # Show emotion distribution
        emotion_counts = {}
        for conv in selected:
            emotion = conv['primary_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("📊 Selected emotion distribution:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count}")
        
        return selected


def main():
    """Example usage of the data loader"""
    
    print("=" * 60)
    print("AMN Data Package Loader - Example Usage")
    print("=" * 60)
    print()
    
    # Initialize loader
    loader = AMNDataLoader(".")  # Current directory
    
    print("\n1️⃣ Loading VAD Lexicon...")
    print("-" * 60)
    lexicon = loader.load_vad_lexicon()
    
    # Test a few words
    test_words = ['happy', 'sad', 'angry', 'calm', 'excited']
    print("\n📝 Sample words:")
    for word in test_words:
        if word in lexicon:
            v, a, d = lexicon[word]
            print(f"  {word}: V={v:+.2f}, A={a:.2f}, D={d:.2f}")
    
    print("\n2️⃣ Loading Conversations...")
    print("-" * 60)
    conversations = loader.load_conversations()
    
    # Show first conversation
    print("\n📖 Sample conversation:")
    conv = conversations[0]
    print(f"  ID: {conv['id']}")
    print(f"  Emotion: {conv['primary_emotion']}")
    print(f"  Turns: {len(conv['turns'])}")
    print(f"\n  First exchange:")
    print(f"    User: {conv['turns'][0]['text']}")
    print(f"    Agent: {conv['turns'][1]['text']}")
    
    print("\n3️⃣ Preparing Experiment Data...")
    print("-" * 60)
    experiment_convos = loader.prepare_for_experiment(n_conversations=30)
    
    # Save for experiments
    output_path = "experiment_conversations.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_convos, f, indent=2)
    print(f"\n✅ Saved {len(experiment_convos)} conversations to {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Data package loaded successfully!")
    print("=" * 60)
    print("\nYou're ready to run experiments! Next steps:")
    print("  1. Update your src/emotion/analyzer.py to use loaded lexicon")
    print("  2. Use experiment_conversations.json for Experiment 1")
    print("  3. Run python scripts/run_experiment.py")
    print()


if __name__ == "__main__":
    main()
