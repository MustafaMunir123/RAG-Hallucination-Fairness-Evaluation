import os
import re
import json
import csv
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
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
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = 'BAAI/bge-reranker-v2-m3'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 4
HYBRID_CANDIDATES = 10

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

def generate(tokenizer, model, prompt, max_new_tokens):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = output[0][inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer


def load_chunks():
    path = os.path.join(KB_DIR, "mitre_attack_kb_combined.txt")
    text = open(path).read()
    techniques = text.split("\n\n==========\n\n")
    chunks = []
    for technique_text in techniques:
        technique_text = technique_text.strip()
        if not technique_text:
            continue
        start = 0
        while start < len(technique_text):
            end = start + CHUNK_SIZE
            chunks.append(technique_text[start:end])
            start = end - CHUNK_OVERLAP
    return chunks

def dense_search(embedder, embeddings, chunks, query, top_k):
    query_embedding = embedder.encode([query])[0]
    scores = []
    for i in range(len(chunks)):
        a = query_embedding
        b = embeddings[i]
        score = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        scores.append(score)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]


def sparse_search(bm25, chunks, query, top_k):
    scores = bm25.get_scores(query.split())
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return ranked[:top_k]

def rrf_merge(ranked_lists, top_k):
    rrf_scores = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)
    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in merged[:top_k]]


def parse_triples(extraction_text):
    triples = []
    for match in re.findall(r"\(([^,()]+),([^,()]+),([^()]+)\)", extraction_text):
        e1, rel, e2 = match
        triples.append((e1.strip(), rel.strip(), e2.strip()))
    return triples

def build_graph(tokenizer, model, chunks):
    graph = nx.Graph()
    entity_to_chunks = {}
    for i, chunk_text in enumerate(chunks):
        extract_prompt = "Extract security entities and their relationships from the text below.\n\n"
        extract_prompt = extract_prompt + "Format each as: (entity1, relationship, entity2)\n\n"
        extract_prompt = extract_prompt + "Example:\n(T1190, is_technique, Exploit Public-Facing Application)\n(Exploit Public-Facing Application, targets, Web Servers)\n\n"
        extract_prompt = extract_prompt + "Text:\n" + chunk_text + "\n\nEntities and relationships:\n"
        extraction_text = generate(tokenizer, model, extract_prompt, 300)
        triples = parse_triples(extraction_text)
        for e1, rel, e2 in triples:
            graph.add_edge(e1, e2, relationship=rel)
            entity_to_chunks.setdefault(e1, set()).add(i)
            entity_to_chunks.setdefault(e2, set()).add(i)
    return graph, entity_to_chunks


def extract_query_entities(tokenizer, model, query):
    prompt = "Extract the key security entities (technique IDs, technique names, software, groups) mentioned or implied in this question.\n\n"
    prompt = prompt + "Question: " + query + "\n\n"
    prompt = prompt + "List one entity per line, no other text:\n"
    extraction_text = generate(tokenizer, model, prompt, 100)
    entities = []
    for line in extraction_text.split("\n"):
        line = line.strip("- ").strip()
        if line:
            entities.append(line)
    return entities

def graph_search(graph, entity_to_chunks, query_entities, top_k):
    related_chunks = set()
    for entity in query_entities:
        matched_node = None
        for node in graph.nodes:
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                matched_node = node
                break
        if matched_node is None:
            continue
        related_chunks.update(entity_to_chunks.get(matched_node, set()))
        for neighbor in graph.neighbors(matched_node):
            related_chunks.update(entity_to_chunks.get(neighbor, set()))
    return list(related_chunks)[:top_k]


def rerank(reranker, chunks, candidate_indices, query, top_k):
    pairs = [[query, chunks[i]] for i in candidate_indices]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidate_indices, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunks[i] for i, score in ranked[:top_k]]
    return top_chunks

def load_questions():
    path = os.path.join(KB_DIR, 'test_questions.json')
    data = json.load(open(path))
    return data


def build_prompt(context_text, question):
    prompt = "You are a network security Q&A assistant. Answer using ONLY the provided context.\n\n"
    prompt = prompt + "Context (from knowledge base):\n" + context_text + "\n\n"
    prompt = prompt + "Question: " + question + "\n\n"
    prompt = prompt + "Instructions:\n"
    prompt = prompt + "- If the context does not contain enough information, say \"I cannot find this in the provided knowledge base.\"\n"
    prompt = prompt + "- Do NOT use external knowledge.\n"
    prompt = prompt + "- Cite which technique IDs from the context support your answer.\n"
    prompt = prompt + "Answer:"
    return prompt

def run_improved():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer, model = load_model()
    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)
    chunks = load_chunks()
    chunk_embeddings = embedder.encode(chunks)
    tokenized_chunks = [c.split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    graph, entity_to_chunks = build_graph(tokenizer, model, chunks)
    print("graph built with", graph.number_of_nodes(), "nodes and", graph.number_of_edges(), "edges")
    questions = load_questions()
    rows = []
    for q in questions:
        dense_ranked = dense_search(embedder, chunk_embeddings, chunks, q["question"], HYBRID_CANDIDATES)
        sparse_ranked = sparse_search(bm25, chunks, q["question"], HYBRID_CANDIDATES)
        query_entities = extract_query_entities(tokenizer, model, q["question"])
        graph_ranked = graph_search(graph, entity_to_chunks, query_entities, HYBRID_CANDIDATES)
        candidate_indices = rrf_merge([dense_ranked, sparse_ranked, graph_ranked], HYBRID_CANDIDATES)
        top_chunks = rerank(reranker, chunks, candidate_indices, q["question"], TOP_K)
        context_text = "\n\n".join(top_chunks)
        prompt = build_prompt(context_text, q["question"])
        answer = generate(tokenizer, model, prompt, 512)
        rows.append([q["id"], q["type"], q["question"], answer, q["category"]])

    out_path = os.path.join(OUTPUT_DIR, "results_improved.csv")
    f = open(out_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["id", "type", "question", "model_answer", "category"])
    for row in rows:
        writer.writerow(row)
    f.close()
    graph_path = os.path.join(OUTPUT_DIR, "entity_graph.gml")
    nx.write_gml(graph, graph_path)
    print("saved", out_path)
    print("saved", graph_path)

run_improved()
