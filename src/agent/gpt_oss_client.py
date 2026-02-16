import requests
import os

def gpt_oss_cloud_chat(prompt, model="gpt-oss:120b-cloud", system_prompt=None, max_tokens=200, temperature=0.7):
    # Example endpoint, update as needed for your actual API
    api_url = os.environ.get("GPT_OSS_CLOUD_API_URL", "http://localhost:8000/api/generate")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    return response.json().get("response", "").strip()
