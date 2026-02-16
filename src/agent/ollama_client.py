
import requests

def ollama_chat(
    prompt,
    model="tinyllama",
    system_prompt=None,
    max_tokens=200,
    temperature=0.7
):
    url = "http://localhost:11434/api/chat"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    response = requests.post(
        url,
        json=payload,
        timeout=30
    )
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
