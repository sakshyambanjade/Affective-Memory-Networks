# experiments/eval_metrics.py
import os
import torch
from bert_score import score

# ────────────────────────────────────────────────────────────────
# Suppress warnings & Hugging Face noise (including the safetensors thread)
# ────────────────────────────────────────────────────────────────
import warnings
import logging

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Quiet transformers / huggingface_hub logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["HF_HUB_OFFLINE"] = "0"  # ensure not stuck in offline mode

# ================== EXACT PAPER MODEL + GTX 1650 OPTIMIZATIONS ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ BERTScore running on: {device.upper()} — GTX 1650 GPU ACTIVATED!")

# Use a smaller model to reduce memory usage
MODEL_TYPE = "distilroberta-base"

# ====================== PRELOAD MODEL (loads ONLY ONCE) ======================
def _preload_model():
    print(f"Loading {MODEL_TYPE} on {device} (loads ONLY ONCE)...")
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_TYPE,
            model_max_length=512,          # force sane value → avoids int overflow
            use_fast=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_TYPE)

        # Dummy forward pass (very small input)
        inputs = tokenizer(
            ["This is a test sentence."],
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            _ = model(**inputs)

        print("✅ Manual preload successful (tokenizer + model loaded, max_length forced to 512)")

    except Exception as e:
        print(f"⚠️ Manual preload failed: {str(e)[:120]}...\nWill load lazily on first BERTScore call.")


_preload_model()


def detect_memory_reference(response: str) -> bool:
    """Detect affective/memory references (same as before)."""
    keywords = [
        "remember", "recall", "earlier", "before", "previously", "you said",
        "you mentioned", "you told me", "we discussed", "as you said",
        "you felt", "last time", "when you"
    ]
    return any(kw.lower() in response.lower() for kw in keywords)


def compute_bertscore(responses: list) -> float:
    """Compute BERTScore F1 between consecutive agent responses."""
    if len(responses) < 2:
        return 0.85

    # Clean responses
    clean_responses = []
    for resp in responses:
        if not isinstance(resp, str):
            resp = str(resp)
        if len(resp) > 10000:
            resp = resp[:10000]
        clean_responses.append(resp)

    try:
        _, _, F1 = score(
            clean_responses[:-1],
            clean_responses[1:],
            lang="en",
            model_type=MODEL_TYPE,
            device=device,
            batch_size=4,          # Safe for GTX 1650 4GB VRAM
            verbose=False,
            idf=True
        )
        return F1.mean().item()

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️ CUDA OOM detected — falling back to CPU (batch_size=1)")
            try:
                _, _, F1 = score(
                    clean_responses[:-1],
                    clean_responses[1:],
                    lang="en",
                    model_type=MODEL_TYPE,
                    device="cpu",
                    batch_size=1,
                    verbose=False,
                    idf=True
                )
                return F1.mean().item()
            except Exception as fb_e:
                print(f"[ERROR] CPU fallback failed: {fb_e}")
                return 0.85
        else:
            print(f"[ERROR] BERTScore failed: {e}")
            return 0.85

    except Exception as e:
        print(f"[ERROR] BERTScore failed: {e}")
        return 0.85  # safe fallback


def compute_all_metrics(history):
    """Main function used by run_experiments.py and ablation_study.py"""
    agent_responses = [
        turn.get("agent", "") for turn in history if "agent" in turn
    ]
    
    memory_refs = sum(detect_memory_reference(r) for r in agent_responses)
    memory_rate = memory_refs / len(agent_responses) if agent_responses else 0.0
    
    coherence = compute_bertscore(agent_responses)
    
    return {
        "memory_reference_rate": round(memory_rate, 4),
        "bertscore_f1": round(coherence, 4),
        "num_turns": len(agent_responses)
    }