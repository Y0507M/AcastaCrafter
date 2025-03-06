import os
import json
import urllib.request
import concurrent.futures

# Downloading real gameplay files for obtain diamond pickaxe.
# https://github.com/openai/Video-Pre-Training?tab=readme-ov-file#contractor-demonstrations
json_file = "all_10xx_Jun_29.json"

with open(json_file, "r") as f:
    data = json.load(f)

base_url = data["basedir"]
file_paths = data["relpaths"]

download_dir = os.path.join(os.getcwd(), "data/Obtain_Diamond_Pickaxe")
os.makedirs(download_dir, exist_ok=True)

total_files = len(file_paths)
downloaded_count = 0

print(f"Total files to download: {total_files}\n", flush=True)

def download_file(file_path):
    """Download a single file and return its status."""
    global downloaded_count
    file_url = base_url + file_path
    local_filename = os.path.join(download_dir, os.path.basename(file_path))

    if os.path.exists(local_filename):
        downloaded_count += 1
        print(f"[{downloaded_count}/{total_files}] Skipped (Already Exists): {local_filename}", flush=True)
        return

    try:
        urllib.request.urlretrieve(file_url, local_filename)
        downloaded_count += 1
        print(f"[{downloaded_count}/{total_files}] Downloaded: {local_filename}", flush=True)
    except Exception as e:
        print(f"Failed to download {file_url}: {e}", flush=True)

# Use ThreadPoolExecutor to download files in parallel
max_workers = min(16, total_files)  # Use up to 16 threads or total files count if less
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(download_file, file_paths)

print(f"\nDownload complete: {downloaded_count}/{total_files} files downloaded successfully.", flush=True)
