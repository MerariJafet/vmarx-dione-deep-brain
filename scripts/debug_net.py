import requests
import os
import traceback

print("ENV HTTP_PROXY:", os.environ.get("HTTP_PROXY"))
print("ENV HTTPS_PROXY:", os.environ.get("HTTPS_PROXY"))

url = "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json"

print(f"\nTesting connection to {url}")
print("1. With default settings...")
try:
    r = requests.head(url, timeout=5)
    print("Status:", r.status_code)
except Exception:
    traceback.print_exc()

print("\n2. With verify=False...")
try:
    r = requests.head(url, verify=False, timeout=5)
    print("Status:", r.status_code)
except Exception:
    traceback.print_exc()
