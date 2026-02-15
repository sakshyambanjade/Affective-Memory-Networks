import requests

def ollama_chat(prompt, model="tinyllama", system_prompt=None, max_tokens=200, temperature=0.7):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    if system_prompt:
        payload["system"] = system_prompt
    response = requests.post(url, json=payload)
    response.raise_for_status()
    return response.json()["response"].strip()
