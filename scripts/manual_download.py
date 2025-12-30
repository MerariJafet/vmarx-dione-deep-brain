import requests
import os
import json
from pathlib import Path
from tqdm import tqdm

# Settings
MODEL_ID = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = Path("models/Mistral-7B-v0.1")
BASE_URL = f"https://huggingface.co/{MODEL_ID}/resolve/main"

# Disable SSL Warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

def get_session():
    s = requests.Session()
    s.verify = False
    return s

def make_request(url):
    s = get_session()
    return s.get(url, allow_redirects=True)

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    out_path = OUTPUT_DIR / filename
    
    if out_path.exists():
        print(f"Skipping {filename}, exists.")
        return

    print(f"Downloading {filename}...")
    try:
        r = get_session().get(url, stream=True)
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        
        with open(out_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        if out_path.exists():
            out_path.unlink()
        raise

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Get File List via API
    api_url = f"https://huggingface.co/api/models/{MODEL_ID}"
    print(f"Fetching file list from {api_url}...")
    r = get_session().get(api_url)
    r.raise_for_status()
    info = r.json()
    
    siblings = info.get("siblings", [])
    files = [s["rfilename"] for s in siblings]
    
    print(f"Found {len(files)} files.")
    
    # Filter for essential files (safetensors, json)
    # Exclude .bin if safetensors exist? Mistral usually has both or just safetensors.
    # We download everything to be safe, except maybe giant pytorch_model.bin if safetensors present.
    
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    
    for f in files:
        if f.endswith(".bin") and has_safetensors:
            continue # Skip bin if we have safetensors
        if f.startswith("."): # Skip git files
            continue
            
        download_file(f)
        
    print("\nDownload Complete.")

if __name__ == "__main__":
    main()
