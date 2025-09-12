from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nyu-visionx/cambrian-34b",
    local_dir="./cambrian-34b",
    max_workers=4  # This controls the number of parallel downloads
)

print("Download complete!")
