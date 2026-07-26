# Baseline RAG Flow

```mermaid
flowchart LR
    A["MITRE ATT&CK KB<br/>5 techniques"] --> B["5 chunks"]
    B --> C["bge-small-en-v1.5<br/>embeddings"]
    Q["20 questions"] --> D["Cosine top-2"]
    C --> D
    D --> E["Qwen2.5-7B<br/>4-bit"]
    E --> F["results_baseline.csv"]
```

# Improved RAG Flow

```mermaid
flowchart LR
    A["MITRE ATT&CK KB<br/>5 techniques"] --> B["116 chunks<br/>500 chars, 100 overlap"]
    B --> C["bge-small-en-v1.5<br/>dense embeddings"]
    B --> D["BM25<br/>sparse index"]
    B --> E["Qwen entity extraction<br/>NetworkX graph"]
    Q["20 questions"] --> F["Dense + BM25 + Graph<br/>candidates"]
    C --> F
    D --> F
    E --> F
    F --> G["RRF merge"]
    G --> H["bge-reranker-v2-m3<br/>top-4"]
    H --> I["Qwen2.5-7B<br/>4-bit"]
    I --> J["results_improved.csv<br/>entity_graph.gml"]
```

# Findings

| Metric | Baseline | Improved | Delta |
|---|---|---|---|
| Overall mean hallucination score (0-3) | 1.15 | 0.5 | ↓ 57% |
| % answers with zero hallucination | 40% | 65% | ↑ 25 pts |
| Fairness std dev across categories | 0.378 | 0.134 | ↓ 65% |
| Worst category (Cloud) mean | 1.75 | 0.25 | ↓ 86% |
| Objective questions mean | 1.2 | 0.3 | ↓ 75% |
| Subjective questions mean | 1.1 | 0.7 | ↓ 36% |

