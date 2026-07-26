# Baseline RAG Flow

```mermaid
flowchart LR
    A["mitre_attack_kb_combined.txt<br/>5 MITRE ATT&CK techniques"] --> B["Split on '==========' delimiter<br/>5 chunks, truncated to 1500 chars"]
    B --> C["Embed chunks<br/>BAAI/bge-small-en-v1.5"]
    C --> D["In-memory numpy vectors<br/>no persisted vector DB"]
    E["test_questions.json<br/>20 questions: 10 objective + 10 subjective"] --> F["Embed question<br/>BAAI/bge-small-en-v1.5"]
    F --> G["Cosine similarity vs all chunks<br/>top_k = 2"]
    D --> G
    G --> H["Build prompt<br/>KB excerpt + question, no instructions"]
    H --> I["Qwen2.5-7B-Instruct<br/>4-bit NF4 quant, best-free-GPU pick"]
    I --> J["results_baseline.csv<br/>id, type, question, model_answer, category"]
```

# Improved RAG Flow

```mermaid
flowchart LR
    A["mitre_attack_kb_combined.txt<br/>5 MITRE ATT&CK techniques"] --> B["Fixed-size chunking<br/>500 chars, 100 char overlap<br/>116 chunks total"]
    B --> C["Embed chunks<br/>BAAI/bge-small-en-v1.5<br/>in-memory numpy vectors"]
    B --> D["BM25Okapi sparse index<br/>tokenized chunk text"]
    B --> E["Graph build: per chunk<br/>Qwen2.5-7B extracts<br/>(entity, relationship, entity) triples"]
    E --> F["NetworkX graph<br/>+ entity-to-chunk map"]
    G["test_questions.json<br/>20 questions"] --> H["Dense search<br/>cosine similarity, top 10"]
    C --> H
    G --> I["Sparse search<br/>BM25Okapi, top 10"]
    D --> I
    G --> J["Query entity extraction<br/>via Qwen2.5-7B"]
    J --> K["Graph search<br/>1-hop neighbor lookup, top 10"]
    F --> K
    H --> L["RRF merge<br/>dense + sparse + graph candidates"]
    I --> L
    K --> L
    L --> M["Cross-encoder rerank<br/>BAAI/bge-reranker-v2-m3<br/>top_k = 4 final chunks"]
    M --> N["Structured prompt<br/>cite technique IDs, no external knowledge,<br/>say 'I cannot find this' if insufficient"]
    N --> O["Qwen2.5-7B-Instruct<br/>4-bit NF4 quant, best-free-GPU pick"]
    O --> P["results_improved.csv<br/>+ entity_graph.gml"]
```
