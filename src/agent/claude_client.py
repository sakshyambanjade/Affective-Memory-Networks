import os
import anthropic

# Claude chat function for Anthropic API

def claude_chat(prompt, system_prompt=None, max_tokens=200, temperature=0.7, model="claude-2.1"):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set.")
    client = anthropic.Anthropic(api_key=api_key)
    system = system_prompt or "You are an emotionally aware agent."
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip() if hasattr(response.content[0], 'text') else response.content[0]['text'].strip()
