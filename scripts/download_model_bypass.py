import os
import requests
import warnings
from huggingface_hub import snapshot_download, configure_http_backend

# 1. Disable Warnings
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# 2. Patch requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# 3. Configure HF Hub to use a session with verify=False
def backend_factory():
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)

# 4. Attempt Download
model_id = "mistralai/Mistral-7B-v0.1"
print(f"Attempting to download {model_id} with SSL verification DISABLED...")

try:
    local_dir = snapshot_download(
        repo_id=model_id,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # Download safetensors or bin, not everything
        resume_download=True
    )
    print(f"SUCCESS: Model downloaded to {local_dir}")
except Exception as e:
    print(f"FAILURE: {e}")
