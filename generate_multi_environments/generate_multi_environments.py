"""
Generate multiple baseline rationale environments (E1–E3)
for IRM training in REV(ECQA) setting.

This version implements:
1. Sequential processing for stability
2. Rate limiting and error handling
3. Cost optimization strategies

E1: Original baseline (leaky)
E2: Leakage masked (regex masking based on answer)  
E3: Counterfactual baseline (Gemini)
"""

import re
import json
import argparse
import os
import time
from pathlib import Path
from tqdm import tqdm
from typing import Dict

import google.generativeai as genai

def generate_antonym(sample: Dict, model, max_retries=3) -> str:
    question = sample.get("question_text", "")
    answer = sample.get("answer_text", "")
    text = sample.get("question_statement_text_masked", "")
    leakage_terms = sample.get("most_leaky_term", [])
    sample_id = sample.get("sample_id", sample.get("id", "unknown"))
    
    prompt = (
        "Please change the masked part of the following sentence with an antonym. \n"
        "Do not change any other part of the sentence except the masked part. \n"
        "Output only a single sentence. \n"
        f"Question context: {question}\n"
        f"Original answer: {answer}\n"
        f"Masked sentence: {text}\n"
        f"the masked term to be changed: {leakage_terms}\n"
        "Antonym version:"
    )
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=256,
                )
            )
            
            # Check if response has valid content
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if candidate.finish_reason == 1:  # STOP - successful completion
                    if candidate.content and candidate.content.parts:
                        result = response.text.strip()
                        return result
                elif candidate.finish_reason == 2:  # SAFETY - content filtered
                    print(f"[BLOCKED] Sample ID {sample_id}: Content filtered by safety, using fallback")
                    print("Using original text as fallback. Please regenerate later if needed.")
                    return text.strip()  # Use original text as fallback
                elif candidate.finish_reason == 3:  # RECITATION - blocked for recitation
                    print(f"[BLOCKED] Sample ID {sample_id}: Content blocked for recitation, using fallback")
                    print("Using original text as fallback. Please regenerate later if needed.")
                    return text.strip()  # Use original text as fallback
                else:  # Other finish reasons (MAX_TOKENS, etc.)
                    raise Exception(f"Generation incomplete, finish_reason: {candidate.finish_reason}")
            else:
                raise Exception("No valid candidates in response")
                
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[FAILED] Sample ID {sample_id}: Generation failed after {max_retries} attempts - {e}")
                print("Using original text as fallback. Please regenerate later if needed.")
                return text.strip()  # Fallback to original
            else:
                print(f"[RETRY] Sample ID {sample_id} attempt {attempt + 1} failed: {e}, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff
    
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["ECQA", "ESNLI"], 
                        help="Task name (ECQA or ESNLI)")
    parser.add_argument("--split_type", type=str, required=True, choices=["train", "val", "test", "synthetic"], 
                        help="Split type (train, val, test, or synthetic)")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--gemini_key", type=str, required=True, help="Gemini API key")
    parser.add_argument("--model_name", type=str, default="gemini-2.5-flash", help="Gemini model name")
    args = parser.parse_args()

    # Configure Gemini API
    genai.configure(api_key=args.gemini_key)

    model = genai.GenerativeModel(
        args.model_name,
        safety_settings=[
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
    )

    # Define paths
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    if args.split_type != 'synthetic':
        input_file = os.path.join(current_path, '..', 'generate_baseline_rationales', 'output', 
                                  args.task, f'baseline_rationales_{args.split_type}_output_ig_t5.jsonl')
    else:
        input_file = os.path.join(current_path, '..', 'generate_baseline_rationales', 'output',  
                                  args.task, f'{args.split_type}_baseline_rationales_output.jsonl')
    
    input_file = os.path.normpath(input_file)
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist!")
        return
    
    print(f"Using input file: {input_file}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            # Add sample ID if not present
            if 'sample_id' not in sample and 'id' not in sample:
                sample['sample_id'] = idx+1
            data.append(sample)
    
    print(f"Total samples to process: {len(data)}")

    # Process data sequentially
    processed_count = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for sample in tqdm(data, desc="Processing samples"):
            q = sample.get("question_text", "")
            a = sample.get("answer_text", "")
            b = sample.get("question_statement_text", "")
            c = sample.get("question_statement_text_masked", "")

            # E1: Original baseline
            E1 = b.strip()
            
            # E2: Masked baseline
            E2 = c.strip()
            
            # E3: Counterfactual baseline
            E3 = generate_antonym(sample, model)

            new_sample = {
                "question_text": q,
                "answer_text": a,
                "baseline": E1,
                "masked": E2,
                "antonym": E3,
            }

            out_f.write(json.dumps(new_sample, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"\nProcessing completed!")
    print(f"Output saved to: {output_path}")
    print(f"Processed {processed_count} samples")


if __name__ == "__main__":
    main()
