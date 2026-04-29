#!/bin/bash
# Usage: ./filter_failed.sh <jobs_file> > <output_file>
# Prints lines from <jobs_file> whose --output-folder lacks a results.json
# This speeds up rerunning
# Make executable: chmod +x filter_failed.sh
# To run: ./filter_failed.sh Mnist_pretrain_TI_jobs.txt > failed_jobs.txt

if [ ! -f "$1" ]; then
    echo "Error: file passed does not exist" >&2
    exit 1
fi

while IFS= read -r line; do
    if [[ $line =~ --output-folder[[:space:]]+([^[:space:]]+) ]]; then
        folder="${BASH_REMATCH[1]}"
        [ -f "$folder/results.json" ] || echo "$line"
    fi
done < "$1"