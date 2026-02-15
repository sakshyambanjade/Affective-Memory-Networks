import os
from itertools import cycle
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

def get_api_key_rotator(env_var_name):
    keys = os.getenv(env_var_name, "").split(",")
    keys = [k.strip() for k in keys if k.strip()]
    if not keys:
        raise RuntimeError(f"No API keys found for {env_var_name}")
    return cycle(keys)
