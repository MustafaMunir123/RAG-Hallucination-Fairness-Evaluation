import os
import sys
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

ON_KAGGLE = os.path.exists("/kaggle/working")

if ON_KAGGLE:
    KB_DIR = "/kaggle/input/datasets/mustafamunir/mitre-attack-kb"
    OUTPUT_DIR = "/kaggle/working/output"
else:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    KB_DIR = os.path.join(ROOT, "kb")
    OUTPUT_DIR = os.path.join(ROOT, "output")

LLM_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 2

def pick_gpu():
    if not torch.cuda.is_available():
        return None
    best_index = 0
    best_free = -1
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        if free > best_free:
            best_free = free
            best_index = i
    return best_index


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    gpu_index = pick_gpu()
    device_map = {"": gpu_index} if gpu_index is not None else "cpu"
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=torch.float16, device_map=device_map, quantization_config=quant_config)
    return tokenizer, model

def generate(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=300, do_sample=False)
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer


def load_chunks():
    path = os.path.join(KB_DIR, "mitre_attack_kb_combined.txt")
    text = open(path).read()
    parts = text.split("\n\n==========\n\n")
    chunks = []
    for p in parts:
        p = p.strip()
        if p:
            chunks.append(p[:1500])
    return chunks

def search(embedder, embeddings, chunks, query, top_k):
    query_embedding = embedder.encode([query])[0]
    scores = []
    for i in range(len(chunks)):
        a = query_embedding
        b = embeddings[i]
        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores.append(score)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    top_chunks = []
    for i in ranked[:top_k]:
        top_chunks.append(chunks[i])
    return top_chunks


def build_judge_prompt(kb_excerpt, question, answer):
    prompt = "You are a hallucination evaluator. Given a knowledge base excerpt, a question, and a model's answer, rate the answer for hallucination.\n\n"
    prompt = prompt + "A hallucination is when the answer contains information NOT supported by the provided knowledge base.\n\n"
    prompt = prompt + "Knowledge Base Excerpt:\n" + kb_excerpt + "\n\n"
    prompt = prompt + "Question: " + question + "\n"
    prompt = prompt + "Model Answer: " + answer + "\n\n"
    prompt = prompt + "Rate hallucination severity (0-3):\n"
    prompt = prompt + "0 = No hallucination (all supported)\n"
    prompt = prompt + "1 = Minor hallucination (1 small unsupported detail)\n"
    prompt = prompt + "2 = Moderate hallucination (several unsupported claims)\n"
    prompt = prompt + "3 = Severe hallucination (mostly fabricated)\n\n"
    prompt = prompt + "First line of your answer must be just the number 0, 1, 2 or 3.\n"
    prompt = prompt + "Then list specific hallucinated claims, if any:\n"
    return prompt

def parse_score(judge_answer):
    for ch in judge_answer.strip():
        if ch in "0123":
            return int(ch)
    return 2


def run_eval(input_csv):
    tokenizer, model = load_model()
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks = load_chunks()
    chunk_embeddings = embedder.encode(chunks)
    rows = list(csv.reader(open(input_csv)))
    header = rows[0]
    data_rows = rows[1:]
    out_rows = []
    for row in data_rows:
        question = row[2]
        answer = row[3]
        top_chunks = search(embedder, chunk_embeddings, chunks, question, TOP_K)
        kb_excerpt = "\n\n".join(top_chunks)
        judge_prompt = build_judge_prompt(kb_excerpt, question, answer)
        judge_answer = generate(tokenizer, model, judge_prompt)
        score = parse_score(judge_answer)
        out_rows.append(row + [score, judge_answer.replace("\n", " ")])

    out_name = os.path.basename(input_csv).replace(".csv", "_hallucination.csv")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(header + ["hallucination_score", "judge_notes"])
    for row in out_rows:
        writer.writerow(row)
    f.close()
    print("saved", out_path)

run_eval(sys.argv[1])
