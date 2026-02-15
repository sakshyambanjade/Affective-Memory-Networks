#!/usr/bin/env python3
"""
Verification script for AMN Data Package
Runs all checks to ensure data integrity
"""

import json
import csv
import sys
from pathlib import Path

def verify_conversations(path):
    """Verify conversation data integrity"""
    print("\n1️⃣ Verifying Conversations...")
    print("-" * 60)
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return False
    
    with open(path, 'r') as f:
        conversations = json.load(f)
    
    # Check count
    if len(conversations) < 100:
        print(f"⚠️  Only {len(conversations)} conversations (expected 100+)")
    else:
        print(f"✅ Found {len(conversations)} conversations")
    
    # Check structure
    required_fields = ['id', 'primary_emotion', 'turns']
    errors = 0
    
    for i, conv in enumerate(conversations[:5]):  # Sample first 5
        for field in required_fields:
            if field not in conv:
                print(f"❌ Conversation {i}: Missing field '{field}'")
                errors += 1
        
        # Check turns
        if 'turns' in conv:
            for turn in conv['turns']:
                if turn['speaker'] not in ['user', 'agent']:
                    print(f"❌ Invalid speaker: {turn['speaker']}")
                    errors += 1
                
                if turn['speaker'] == 'user':
                    if 'valence' not in turn or 'arousal' not in turn:
                        print(f"❌ User turn missing VAD scores")
                        errors += 1
                    else:
                        # Validate ranges
                        v = turn['valence']
                        a = turn['arousal']
                        d = turn.get('dominance', 0.5)
                        
                        if not (-1 <= v <= 1):
                            print(f"❌ Valence out of range: {v}")
                            errors += 1
                        if not (0 <= a <= 1):
                            print(f"❌ Arousal out of range: {a}")
                            errors += 1
                        if not (0 <= d <= 1):
                            print(f"❌ Dominance out of range: {d}")
                            errors += 1
    
    if errors == 0:
        print("✅ All conversation checks passed")
        return True
    else:
        print(f"❌ Found {errors} errors in conversations")
        return False

def verify_lexicon(path):
    """Verify VAD lexicon integrity"""
    print("\n2️⃣ Verifying VAD Lexicon...")
    print("-" * 60)
    
    if not path.exists():
        print(f"❌ File not found: {path}")
        return False
    
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Check count
    if len(rows) < 200:
        print(f"⚠️  Only {len(rows)} words (expected 200+)")
    else:
        print(f"✅ Found {len(rows)} words in lexicon")
    
    # Check structure
    required_cols = ['Word', 'Valence', 'Arousal', 'Dominance']
    if not all(col in rows[0] for col in required_cols):
        print(f"❌ Missing required columns")
        return False
    
    # Validate ranges
    errors = 0
    for i, row in enumerate(rows[:10]):  # Sample first 10
        try:
            v = float(row['Valence'])
            a = float(row['Arousal'])
            d = float(row['Dominance'])
            
            if not (-1 <= v <= 1):
                print(f"❌ Row {i}: Valence out of range: {v}")
                errors += 1
            if not (0 <= a <= 1):
                print(f"❌ Row {i}: Arousal out of range: {a}")
                errors += 1
            if not (0 <= d <= 1):
                print(f"❌ Row {i}: Dominance out of range: {d}")
                errors += 1
        except ValueError as e:
            print(f"❌ Row {i}: Invalid numeric value: {e}")
            errors += 1
    
    # Check word diversity
    words = [row['Word'].lower() for row in rows]
    if len(words) != len(set(words)):
        print(f"⚠️  Duplicate words found")
    
    # Check coverage of common emotion words
    emotion_words = ['happy', 'sad', 'angry', 'afraid', 'excited', 'calm']
    missing = [w for w in emotion_words if w not in words]
    if missing:
        print(f"⚠️  Missing common words: {missing}")
    else:
        print(f"✅ All common emotion words present")
    
    if errors == 0:
        print("✅ All lexicon checks passed")
        return True
    else:
        print(f"❌ Found {errors} errors in lexicon")
        return False

def verify_structure():
    """Verify directory structure"""
    print("\n3️⃣ Verifying Package Structure...")
    print("-" * 60)
    
    base = Path(".")
    required_dirs = ['conversations', 'lexicons', 'test_cases', 'scripts']
    required_files = [
        'README.md',
        'conversations/empathetic_dialogues_100.json',
        'lexicons/warriner_vad_2000.csv',
        'test_cases/TEST_CASES.md',
        'scripts/load_data.py'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        if not (base / dir_name).exists():
            print(f"❌ Missing directory: {dir_name}")
            all_good = False
        else:
            print(f"✅ Directory exists: {dir_name}")
    
    for file_path in required_files:
        if not (base / file_path).exists():
            print(f"❌ Missing file: {file_path}")
            all_good = False
    
    if all_good:
        print("✅ All required files and directories present")
    
    return all_good

def main():
    print("=" * 60)
    print("AMN Data Package Verification")
    print("=" * 60)
    
    # Run all checks
    structure_ok = verify_structure()
    conversations_ok = verify_conversations(Path("conversations/empathetic_dialogues_100.json"))
    lexicon_ok = verify_lexicon(Path("lexicons/warriner_vad_2000.csv"))
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    results = {
        "Package Structure": structure_ok,
        "Conversations Data": conversations_ok,
        "VAD Lexicon": lexicon_ok
    }
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")
    
    if all(results.values()):
        print("\n🎉 All checks passed! Data package is ready to use.")
        print("\nNext steps:")
        print("  1. python scripts/load_data.py")
        print("  2. Integrate with your AMN project")
        print("  3. Run experiments!")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
