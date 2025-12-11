#!/bin/bash

# Simple script to run the optimized baseline environment generation
# Usage: ./run_optimized_batch.sh [task] [split_type] [gemini_key]

set -e  # Exit on any error

# Default values
TASK=${1:-"ECQA"}
SPLIT_TYPE=${2:-"train"}
GEMINI_KEY="AIzaSyDD9H332B0FPcP0kfCFZkwHpCh1TaoAJxc"
MODEL_NAME="gemini-2.5-flash-lite"


# Check if Gemini API key is provided
if [ -z "$GEMINI_KEY" ]; then
    echo "Error: Gemini API key is required!"
    echo "Usage: $0 [task] [split_type] [gemini_key]"
    echo "Example: $0 ECQA train your_gemini_api_key"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"

# Generate output filename
OUTPUT_FILE="$OUTPUT_DIR/multi_environments_${TASK}_${SPLIT_TYPE}.jsonl"

echo "Starting multi environment generation..."
echo "Task: $TASK"
echo "Split type: $SPLIT_TYPE"
echo "Output file: $OUTPUT_FILE"
echo "Model: $MODEL_NAME"
echo ""

# Run the Python script
python generate_multi_environments_old.py \
    --task "$TASK" \
    --split_type "$SPLIT_TYPE" \
    --output "$OUTPUT_FILE" \
    --gemini_key "$GEMINI_KEY" \
    --model_name "$MODEL_NAME"

echo ""
echo "Multiple environment generation completed!"
echo "Output saved to: $OUTPUT_FILE"

# Optional: Show file size and line count
if [ -f "$OUTPUT_FILE" ]; then
    echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
    echo "Number of lines: $(wc -l < "$OUTPUT_FILE")"
fi
