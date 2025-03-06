#!/bin/bash
#SBATCH -A cs175_class
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --partition=standard
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --job-name=download_minerl-v0_dataset

# Create the data directory if it doesn't exist
mkdir -p ./data
mkdir -p ./minerl-v0_dataset_download_logs
cd ./data

# List of direct download links
declare -A FILE_URLS
FILE_URLS=(
    ["MineRLNavigate-v0.zip"]="https://zenodo.org/record/12659939/files/MineRLNavigate-v0.zip"
    ["MineRLObtainDiamond-v0.zip"]="https://zenodo.org/record/12659939/files/MineRLObtainDiamond-v0.zip"
    ["MineRLObtainIronPickaxe-v0.zip"]="https://zenodo.org/record/12659939/files/MineRLObtainIronPickaxe-v0.zip"
    ["MineRLTreechop-v0.zip"]="https://zenodo.org/record/12659939/files/MineRLTreechop-v0.zip"
)

# Loop through URLs and download in parallel, logging separately
for file in "${!FILE_URLS[@]}"; do
    wget -c --progress=dot:giga "${FILE_URLS[$file]}" > "../minerl-v0_dataset_download_logs/${file}.log" 2>&1 &
done

# Wait for all background downloads to finish
wait

echo "Download completed. Logs are in ./minerl-v0_dataset_download_logs/"
