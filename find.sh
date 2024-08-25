#!/bin/bash

# Initialize an empty array to store matching directories
matching_directories=()

# Define the expected values
vae_directory="train_xl/train_150/"
vae_id="22326124"
ldm_directory="train_xl/train_50/"
ldm_id="22342690"

# Function to check the JSON file for the required conditions
check_json() {
  json_file="$1"

  # Extract values using jq
  vae_directory_value=$(jq -r '.jobs.vae.directory' "$json_file")
  vae_id_value=$(jq -r '.jobs.vae.id' "$json_file")
  ldm_directory_value=$(jq -r '.jobs.ldm.directory' "$json_file")
  ldm_id_value=$(jq -r '.jobs.ldm.id' "$json_file")

  # Compare the extracted values with the expected ones
  if [[ "$vae_directory_value" == "$vae_directory" ]] && \
     [[ "$vae_id_value" == "$vae_id" ]] && \
     [[ "$ldm_directory_value" == "$ldm_directory" ]] && \
     [[ "$ldm_id_value" == "$ldm_id" ]]; then
    return 0
  else
    return 1
  fi
}

# Recursively find all JSON files and process them
while IFS= read -r -d '' json_file; do
  if check_json "$json_file"; then
    # Get the directory containing the JSON file and add it to the list
    dir=$(dirname "$json_file")
    matching_directories+=("$dir")
  fi
done < <(find . -type f -name "*.json" -print0)

# Print the list of matching directories
if [[ ${#matching_directories[@]} -gt 0 ]]; then
  echo "Matching directories:"
  for dir in "${matching_directories[@]}"; do
    echo "$dir"
  done
else
  echo "No matching directories found."
fi
