import pandas as pd
import torch
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def contains_answer(rationale: str, answer: str) -> bool:
    """
    檢查 rationale 是否包含 answer（大小寫不敏感，簡單 substring）。
    之後如果要更嚴格可以改成 token-level。
    """
    if answer is None:
        return True
    r = (rationale or "").lower()
    a = (answer or "").lower().strip()
    if not a:
        return True
    return a in r
    
def regenerate_with_stronger_decoding(input_text: str,
                                      tokenizer,
                                      model,
                                      device,
                                      max_length: int = 128) -> str:
    """
    用 num_beams=5 對同一個 input_text 再生成一次。
    不在這裡檢查 answer，純粹負責重生。
    """
    enc = tokenizer(
        [input_text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            num_beams=5,           # 比第一次的 beam=1 強
            max_length=max_length
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

def batch(iterable, n=16):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

MODEL_DIR = '/media/vchd/新增磁碟區/Neil/Research/generate_baseline_rationales/prepare_baseline_rationale/ECQA/seq2seq_converter/model_data/question-converter-t5-3b'

# 準備模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
elapsed_time = time.time() - start_time
print(f"Model loaded in {elapsed_time:.2f} seconds, using device: {device}")
model.eval()

SPLIT = ["train", "val", "test"]
os.makedirs('../../generated_baseline_rationales', exist_ok=True)

for split in SPLIT:
    INPUT_CSV = f'/media/vchd/新增磁碟區/Neil/Research/dataset/ecqa/output/ecqa_{split}.csv'
    OUTPUT_JSONL = f'/media/vchd/新增磁碟區/Neil/Research/generated_baseline_rationales/output/ECQA/baseline_rationales_{split}_output.jsonl'

    if not os.path.exists(INPUT_CSV):
        print(f"[SKIP] Input CSV not found: {INPUT_CSV}")
        continue
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    # 讀資料
    df = pd.read_csv(INPUT_CSV)
    questions = df["q_text"].astype(str).tolist()
    answers = df["q_ans"].astype(str).tolist()
 
    total_missing_first_pass = 0              # 第一次生成就缺 answer 的數量
    total_still_missing_after_regen = 0       # 5 次內都沒成功的數量
    problematic_sample_indices = []           # 記錄「還是缺 answer」的 sample index（1-based）

    inputs = [f"{q} </s> {a}" for q, a in zip(questions, answers)]
    pairs = list(zip(inputs, answers))   # (input_text, answer) 成對，方便對齊
    gens = []
    max_attempts = 5
    with torch.no_grad():
        # 這裡的 idx 是 global index（0-based），方便你對應回原始資料
        for chunk in tqdm(list(batch(list(enumerate(pairs)), 16)),
                        total=(len(pairs) + 15)//16,
                        desc=f"Generating rationales (up to {max_attempts} attempts)"):

            # chunk: list of (global_idx, (input_text, answer))
            chunk_indices = [item[0] for item in chunk]
            chunk_inputs  = [item[1][0] for item in chunk]
            chunk_answers = [item[1][1] for item in chunk]

            # 第一次生成：完全照你原本邏輯（num_beams=1）
            enc = tokenizer(
                chunk_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)

            out = model.generate(**enc, num_beams=1, max_length=128)
            texts = tokenizer.batch_decode(out, skip_special_tokens=True)

            fixed_texts = []
            for local_i, (text, inp, ans, global_idx) in enumerate(
                zip(texts, chunk_inputs, chunk_answers, chunk_indices)
            ):
                attempts = 1
                current_text = text

                # # 如果第一次就缺 answer → 記一筆，然後開始重生
                # if not contains_answer(current_text, ans):
                #     total_missing_first_pass += 1

                #     while attempts < max_attempts and not contains_answer(current_text, ans):
                #         attempts += 1
                #         current_text = regenerate_with_stronger_decoding(
                #             input_text=inp,
                #             tokenizer=tokenizer,
                #             model=model,
                #             device=device,
                #             max_length=128
                #         )

                #     # 走出 while 之後再檢查一次
                #     if not contains_answer(current_text, ans):
                #         total_still_missing_after_regen += 1
                #         # global_idx 是 0-based，+1 變成「第幾個 sample」
                #         sample_id_1based = global_idx + 1
                #         problematic_sample_indices.append(sample_id_1based)

                fixed_texts.append(current_text)

            gens.extend(fixed_texts)

    # -------------------------------------------------
    # 這裡開始做「刪除有問題 sample」＋「寫 filtered CSV」
    # -------------------------------------------------
    n_total = len(df)
    # bad_set = set(problematic_sample_indices)
    # good_indices = [i for i in range(n_total) if i not in bad_set]

    print(f"Total samples: {n_total}")
    # print(f"First-pass missing-answer count: {total_missing_first_pass}")
    # print(f"Still-missing-after-{max_attempts}-attempts count: {total_still_missing_after_regen}")
    # print(f"Filtered out {len(bad_set)} samples; kept {len(good_indices)} samples.")

    # if bad_set:
    #     print("Filtered-out (0-based) indices:", sorted(bad_set))
    # else:
    #     print("All samples contain answer token within the allowed attempts.")

    # 1) 產生「只保留 good_indices」的 baseline rationale JSONL
    # filtered_questions = [questions[i] for i in good_indices]
    # filtered_answers   = [answers[i] for i in good_indices]
    # filtered_gens      = [gens[i] for i in good_indices]

    filtered_questions = questions
    filtered_answers   = answers
    filtered_gens      = gens

    out_df = pd.DataFrame({
        "question_text": filtered_questions,
        "answer_text": filtered_answers,
        "question_statement_text": filtered_gens
    })
    out_df.to_json(OUTPUT_JSONL, lines=True, orient="records", force_ascii=False)
    print(f"Filtered baseline rationales saved to -> {OUTPUT_JSONL}")

    # 2) 複製一份 INPUT_CSV，但刪掉 bad samples
    # filtered_df = df.iloc[good_indices].reset_index(drop=True)
    # filtered_csv_path = INPUT_CSV.replace(".csv", "_filtered.csv")
    # filtered_df.to_csv(filtered_csv_path, index=False, encoding="utf-8")
    # print(f"Filtered INPUT_CSV saved to -> {filtered_csv_path}")
