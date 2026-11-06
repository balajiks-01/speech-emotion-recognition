import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "Audio_Speech_Actors_01-24"
HF_API = "https://huggingface.co/api/datasets/MahiA/RAVDESS"
RAW_BASE_URL = "https://huggingface.co/datasets/MahiA/RAVDESS/resolve/main/"

TARGET_EMOTIONS = {"02", "03", "06", "07"}

def get_target_file_list():
    print("Fetching dataset metadata from repository...")
    r = requests.get(HF_API)
    r.raise_for_status()
    all_files = [s["rfilename"] for s in r.json().get("siblings", []) if s["rfilename"].endswith(".wav")]
    
    # Filter for audio-only speech: '03-01-'
    target_files = []
    for f in all_files:
        fname = f.split("/")[-1]
        parts = fname.split("-")
        if len(parts) == 7 and parts[0] == "03" and parts[1] == "01" and parts[2] in TARGET_EMOTIONS:
            target_files.append((f, fname, parts[6])) # (repo_rel_path, filename, actor_id)
            
    print(f"Identified {len(target_files)} target audio files (expected 768).")
    return target_files

def download_file(item):
    repo_path, fname, actor_id = item
    actor_dir = DATA_DIR / f"Actor_{actor_id}"
    actor_dir.mkdir(parents=True, exist_ok=True)
    out_file = actor_dir / fname
    
    if out_file.exists() and out_file.stat().st_size > 1000:
        return True
        
    url = RAW_BASE_URL + repo_path
    res = requests.get(url, timeout=30)
    if res.status_code == 200 and len(res.content) > 1000:
        with open(out_file, "wb") as f:
            f.write(res.content)
        return True
    else:
        raise RuntimeError(f"Failed to download {url}: Status {res.status_code}")

def main():
    target_files = get_target_file_list()
    assert len(target_files) == 768, f"Expected 768 files, got {len(target_files)}"
    
    print(f"Downloading 768 files into {DATA_DIR} using 16 concurrent workers...")
    completed = 0
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_file, item): item for item in target_files}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 100 == 0 or completed == 768:
                print(f"Downloaded {completed}/768 files...")
                
    print(f"All 768 RAVDESS audio files successfully downloaded to {DATA_DIR}!")

if __name__ == "__main__":
    main()
