import os
import json
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

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL = 'BAAI/bge-small-en-v1.5'
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
    print("model loaded on gpu", gpu_index)
    return tokenizer, model

def generate(tokenizer, model, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
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

def embed_chunks(embedder, chunks):
    embeddings = embedder.encode(chunks)
    return embeddings


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


def load_questions():
    path = os.path.join(KB_DIR, 'test_questions.json')
    data = json.load(open(path))
    return data

def build_prompt(context_text, question):
    prompt = "You are a network security chatbot. Use the following knowledge base to answer the question.\n\n"
    prompt = prompt + "Knowledge Base:\n" + context_text + "\n\n"
    prompt = prompt + "Question: " + question + "\nAnswer:"
    return prompt


def run_baseline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer, model = load_model()
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks = load_chunks()
    chunk_embeddings = embed_chunks(embedder, chunks)
    questions = load_questions()
    rows = []
    for q in questions:
        top_chunks = search(embedder, chunk_embeddings, chunks, q["question"], TOP_K)
        context_text = "\n\n".join(top_chunks)
        prompt = build_prompt(context_text, q["question"])
        answer = generate(tokenizer, model, prompt)
        rows.append([q["id"], q["type"], q["question"], answer, q["category"]])

    out_path = os.path.join(OUTPUT_DIR, "results_baseline.csv")
    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["id", "type", "question", "model_answer", "category"])
    for row in rows:
        writer.writerow(row)
    f.close()
    print("saved", out_path)

run_baseline()
