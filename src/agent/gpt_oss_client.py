import requests
import os

def gpt_oss_cloud_chat(prompt, model="gpt-oss:120b-cloud", system_prompt=None, max_tokens=200, temperature=0.7):
    # Use Ollama's chat endpoint by default
    api_url = os.environ.get("GPT_OSS_CLOUD_API_URL", "http://localhost:11434/api/chat")
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
    response = requests.post(api_url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()
