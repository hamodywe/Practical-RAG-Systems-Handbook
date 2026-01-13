# The Practical RAG Systems Handbook
> *A comprehensive guide to building production-ready Retrieval-Augmented Generation systems.*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 📖 Overview

**The Practical RAG Systems Handbook** is a deep-dive technical resource for **intermediate to advanced AI engineers** who want to design, build, and deploy robust **Retrieval-Augmented Generation (RAG)** systems in production environments.

This handbook covers:
- Core RAG architecture and theory
- Chunking, embedding, and indexing strategies
- Vector database selection and optimization
- Advanced techniques: query rewriting, re-ranking, and hybrid search
- Evaluation frameworks and metrics
- Common pitfalls and anti-patterns
- Production best practices and scaling considerations

---

## 📚 Table of Contents

1.  [What is RAG?](#1-what-is-rag)
2.  [RAG Architecture](#2-rag-architecture)
3.  [Chunking Strategies](#3-chunking-strategies)
4.  [Embedding Models Comparison](#4-embedding-models-comparison)
5.  [Vector Databases](#5-vector-databases)
6.  [Query Rewriting & Re-Ranking](#6-query-rewriting--re-ranking)
7.  [Evaluation Metrics for RAG](#7-evaluation-metrics-for-rag)
8.  [Common RAG Mistakes & Anti-Patterns](#8-common-rag-mistakes--anti-patterns)
9.  [Production Best Practices](#9-production-best-practices)
10. [Tools & Frameworks](#10-tools--frameworks)
11. [References](#11-references)

---

## 1. What is RAG?

### Simple Explanation

**RAG (Retrieval-Augmented Generation)** is a technique that makes Large Language Models (LLMs) smarter by giving them access to external knowledge. Instead of relying solely on what the model "memorized" during training, RAG retrieves relevant documents from a database and injects them into the prompt, allowing the LLM to generate answers grounded in real, up-to-date information.

> **Analogy:** Imagine an open-book exam. The LLM is the student, and RAG gives them access to a textbook (your knowledge base) during the test.

### Technical Explanation

RAG decouples **parametric knowledge** (frozen in model weights) from **non-parametric knowledge** (stored in an external retrieval index). At inference time:

1.  A user query `q` is transformed into an embedding `e_q` using an encoder model.
2.  The embedding is used to perform a similarity search (e.g., cosine similarity, dot product) against a pre-indexed corpus of document chunks `{d_1, d_2, ..., d_n}`.
3.  The top-k most similar chunks `{d_i, d_j, ..., d_k}` are retrieved.
4.  These chunks are concatenated with the original query to form an augmented prompt.
5.  The augmented prompt is passed to a generative LLM, which produces a grounded response.

This architecture addresses key LLM limitations:
- **Knowledge cutoff:** LLMs don't know about events after their training date.
- **Hallucinations:** Grounding in retrieved documents reduces fabricated answers.
- **Domain specificity:** RAG enables LLMs to answer questions about private or specialized corpora.

---

## 2. RAG Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAG PIPELINE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ┌──────────────┐                                                              │
│   │   User Query │                                                              │
│   └──────┬───────┘                                                              │
│          │                                                                      │
│          ▼                                                                      │
│   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐       │
│   │  Query Processor │ ──▶ │  Embedding Model │ ──▶ │   Query Vector   │       │
│   │  (Rewrite/Expand)│     │  (e.g., E5, BGE) │     │   e_q ∈ ℝ^d      │       │
│   └──────────────────┘     └──────────────────┘     └────────┬─────────┘       │
│                                                               │                 │
│          ┌────────────────────────────────────────────────────┘                 │
│          │                                                                      │
│          ▼                                                                      │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                      VECTOR DATABASE                             │          │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐            │          │
│   │  │ Chunk 1  │ │ Chunk 2  │ │ Chunk 3  │ │ Chunk N  │  ...       │          │
│   │  │  e_1     │ │  e_2     │ │  e_3     │ │  e_n     │            │          │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘            │          │
│   └──────────────────────────────┬───────────────────────────────────┘          │
│                                  │ Top-K Similarity Search                      │
│                                  ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                      RETRIEVED CHUNKS                            │          │
│   │  [Chunk 7] [Chunk 23] [Chunk 101] ... (ranked by similarity)     │          │
│   └──────────────────────────────┬───────────────────────────────────┘          │
│                                  │                                              │
│                                  ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                      RE-RANKER (Optional)                        │          │
│   │  Cross-encoder model rescores chunks for relevance              │          │
│   └──────────────────────────────┬───────────────────────────────────┘          │
│                                  │                                              │
│                                  ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                      PROMPT CONSTRUCTION                         │          │
│   │  System Prompt + Retrieved Context + User Query                  │          │
│   └──────────────────────────────┬───────────────────────────────────┘          │
│                                  │                                              │
│                                  ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                      GENERATIVE LLM                              │          │
│   │  (GPT-4, Claude, Gemini, Llama, Mistral, etc.)                   │          │
│   └──────────────────────────────┬───────────────────────────────────┘          │
│                                  │                                              │
│                                  ▼                                              │
│   ┌──────────────┐                                                              │
│   │   Response   │                                                              │
│   └──────────────┘                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Indexing Pipeline (Offline)

Before the runtime RAG loop, documents must be preprocessed and indexed:

```
┌────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────────┐
│  Raw Docs  │ ──▶ │  Chunking  │ ──▶ │ Embedding  │ ──▶ │ Vector Index   │
│ (PDF, TXT) │     │ Strategy   │     │   Model    │     │ (FAISS, etc.)  │
└────────────┘     └────────────┘     └────────────┘     └────────────────┘
```

---

## 3. Chunking Strategies

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval. The strategy you choose significantly impacts retrieval quality.

### 3.1 Fixed-Size Chunking

Splits documents into chunks of a fixed number of tokens or characters, often with overlap.

| Parameter     | Typical Value  |
| :------------ | :------------- |
| Chunk Size    | 256–1024 tokens |
| Overlap       | 10–20% of chunk size |

**Pros:** Simple, predictable, fast.
**Cons:** May split sentences or paragraphs mid-thought, losing semantic coherence.

```python
# Pseudo-code: Fixed-size chunking
def fixed_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

### 3.2 Semantic Chunking

Splits documents at natural semantic boundaries (sentences, paragraphs, sections).

**Pros:** Preserves meaning, better retrieval quality for complex queries.
**Cons:** Variable chunk sizes, requires NLP parsing (e.g., spaCy, NLTK).

```python
# Pseudo-code: Sentence-based semantic chunking
import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunk(text: str, max_chunk_tokens: int = 512) -> list[str]:
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks, current_chunk = [], ""
    for sent in sentences:
        if len(current_chunk) + len(sent) < max_chunk_tokens:
            current_chunk += " " + sent
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sent
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks
```

### 3.3 Hybrid Chunking

Combines fixed-size limits with semantic boundaries. Respects sentence/paragraph breaks but enforces a maximum token count.

**Best of both worlds:** Semantic coherence + predictable memory usage.

```python
# Pseudo-code: Hybrid chunking with LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
)
chunks = splitter.split_text(document_text)
```

### Chunking Strategy Comparison

| Strategy       | Semantic Coherence | Size Predictability | Implementation Complexity |
| :------------- | :----------------- | :------------------ | :------------------------ |
| Fixed-Size     | ⭐⭐                  | ⭐⭐⭐⭐⭐                 | ⭐ (Low)                   |
| Semantic       | ⭐⭐⭐⭐⭐               | ⭐⭐                   | ⭐⭐⭐ (Medium)               |
| Hybrid         | ⭐⭐⭐⭐                | ⭐⭐⭐⭐                  | ⭐⭐ (Low-Medium)            |

---

## 4. Embedding Models Comparison

Embedding models convert text into dense vector representations. Model choice affects retrieval accuracy, latency, and cost.

| Model                        | Dimensions | Context Length | Strengths                                  | Use Case                       |
| :--------------------------- | :--------- | :------------- | :----------------------------------------- | :----------------------------- |
| **OpenAI `text-embedding-3-large`** | 3072       | 8191 tokens    | High quality, easy API                     | General purpose, production    |
| **OpenAI `text-embedding-3-small`** | 1536       | 8191 tokens    | Cheaper, still high quality                | Cost-sensitive production      |
| **Cohere `embed-english-v3.0`**     | 1024       | 512 tokens     | Excellent for English, search-optimized    | English-only, search           |
| **Google `textembedding-gecko`**    | 768        | 2048 tokens    | GCP native, good multilingual              | GCP-integrated apps            |
| **BGE-Large (BAAI)**         | 1024       | 512 tokens     | Open-source, top MTEB leaderboard          | Self-hosted, privacy-sensitive |
| **E5-Large-v2 (Microsoft)**  | 1024       | 512 tokens     | Strong open-source, instruction-tuned      | Self-hosted, versatile         |
| **Sentence-Transformers**    | 384–1024   | 256–512 tokens | Wide variety, easy to fine-tune            | Research, prototyping          |
| **Jina AI `jina-embeddings-v2`**    | 768        | 8192 tokens    | Long-context, open-source                  | Long documents                 |

### Key Considerations

1.  **Dimensionality:** Higher dimensions = more expressive but more storage/compute.
2.  **Context Length:** Ensure it matches your chunk size.
3.  **Domain Fit:** General models may underperform on specialized domains; consider fine-tuning.
4.  **Latency:** API-based models add network overhead; self-hosted models reduce latency.

```python
# Pseudo-code: Generating embeddings with OpenAI
from openai import OpenAI
client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding
```

---

## 5. Vector Databases

Vector databases store and index embeddings for fast similarity search. Here's a comparison of leading options:

| Database       | Type              | Scalability    | Key Features                                      | Best For                         |
| :------------- | :---------------- | :------------- | :------------------------------------------------ | :------------------------------- |
| **FAISS**      | Library (In-memory) | Medium         | Blazing fast, GPU support, open-source            | Prototyping, single-node apps    |
| **Pinecone**   | Managed SaaS      | High           | Fully managed, real-time updates, metadata filters | Production, zero-ops             |
| **Weaviate**   | Open-source / Cloud | High           | GraphQL API, hybrid search, ML module integrations | Hybrid keyword + vector search   |
| **Qdrant**     | Open-source / Cloud | High           | Rust-based, fast, rich filtering, gRPC support    | High-performance self-hosted     |
| **Milvus**     | Open-source       | Very High      | Distributed, cloud-native, GPU acceleration       | Large-scale enterprise           |
| **Chroma**     | Open-source (Lightweight) | Low-Medium | Python-native, embedded, simple API               | Prototyping, local development   |
| **pgvector**   | PostgreSQL Extension | Medium       | SQL-native, no new infra needed                   | Teams already on PostgreSQL      |

### FAISS Example

```python
# Pseudo-code: FAISS indexing and search
import faiss
import numpy as np

# Assume `embeddings` is an np.array of shape (n_docs, dim)
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)  # L2 distance
index.add(embeddings)

# Search
query_vector = np.array([get_embedding("What is RAG?")])
distances, indices = index.search(query_vector, k=5)  # Top 5
```

### Pinecone Example

```python
# Pseudo-code: Pinecone upsert and query
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")
index = pc.Index("my-rag-index")

# Upsert vectors
index.upsert(vectors=[
    {"id": "doc1", "values": embedding_1, "metadata": {"source": "manual.pdf"}},
    {"id": "doc2", "values": embedding_2, "metadata": {"source": "faq.txt"}},
])

# Query
results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

---

## 6. Query Rewriting & Re-Ranking

### 6.1 Query Rewriting

Raw user queries are often vague, misspelled, or poorly structured. Query rewriting improves retrieval by transforming the query before embedding.

**Techniques:**
- **Expansion:** Add synonyms or related terms.
- **Decomposition:** Break a complex query into sub-queries.
- **Hypothetical Document Embedding (HyDE):** Ask the LLM to generate a hypothetical answer, embed *that*, and use it for retrieval.

```python
# Pseudo-code: HyDE query rewriting
def hyde_rewrite(query: str, llm) -> str:
    prompt = f"Write a short, factual paragraph that answers the following question:\n{query}"
    hypothetical_doc = llm.generate(prompt)
    return hypothetical_doc  # Embed this instead of the raw query
```

### 6.2 Re-Ranking

Initial retrieval (bi-encoder) prioritizes speed over precision. Re-ranking uses a more powerful **cross-encoder** model to rescore the top-k candidates.

| Stage         | Model Type   | Speed   | Accuracy |
| :------------ | :----------- | :------ | :------- |
| Retrieval     | Bi-Encoder   | Fast    | Good     |
| Re-Ranking    | Cross-Encoder | Slower  | Excellent |

**Popular Re-Rankers:**
- Cohere `rerank-english-v2.0`
- BGE Reranker (BAAI)
- ColBERT
- Sentence-Transformers Cross-Encoders

```python
# Pseudo-code: Cohere re-ranking
import cohere
co = cohere.Client("YOUR_API_KEY")

def rerank(query: str, documents: list[str], top_n: int = 3) -> list[str]:
    response = co.rerank(
        query=query,
        documents=documents,
        top_n=top_n,
        model="rerank-english-v2.0"
    )
    return [doc.document["text"] for doc in response.results]
```

---

## 7. Evaluation Metrics for RAG

RAG systems require evaluation at two levels: **Retrieval Quality** and **Generation Quality**.

### 7.1 Retrieval Metrics

| Metric            | Description                                                  |
| :---------------- | :----------------------------------------------------------- |
| **Recall@K**      | % of relevant documents retrieved in the top K results.       |
| **Precision@K**   | % of top K results that are relevant.                         |
| **MRR (Mean Reciprocal Rank)** | Average of 1/rank for the first relevant result.  |
| **NDCG (Normalized Discounted Cumulative Gain)** | Measures ranking quality, weighted by position. |

### 7.2 Generation Metrics

| Metric            | Description                                                                 | Framework      |
| :---------------- | :-------------------------------------------------------------------------- | :------------- |
| **Faithfulness**  | Does the answer use *only* information from the retrieved context? (No hallucination) | RAGAS, TruLens |
| **Answer Relevance** | Is the answer relevant to the user's question?                           | RAGAS, TruLens |
| **Context Relevance** | Are the retrieved chunks relevant to the question?                       | RAGAS          |
| **Context Recall** | Does the retrieved context contain the ground-truth answer?                | RAGAS          |

### RAGAS Example

```python
# Pseudo-code: RAGAS evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

# `results` is a Dataset with columns: question, answer, contexts, ground_truth
scores = evaluate(
    results,
    metrics=[faithfulness, answer_relevancy, context_precision]
)
print(scores)
# {'faithfulness': 0.92, 'answer_relevancy': 0.88, 'context_precision': 0.85}
```

---

## 8. Common RAG Mistakes & Anti-Patterns

### ❌ Anti-Pattern 1: One-Size-Fits-All Chunking
**Mistake:** Using default 500-token fixed chunks for all document types.
**Fix:** Analyze your documents. Code needs different chunking than legal contracts. Use semantic or hybrid strategies.

### ❌ Anti-Pattern 2: Ignoring Metadata
**Mistake:** Indexing only text embeddings, discarding source, date, and category metadata.
**Fix:** Store metadata in your vector DB. Use it for filtering (e.g., "only search documents from 2024").

### ❌ Anti-Pattern 3: Retrieving Too Many (or Too Few) Chunks
**Mistake:** Stuffing 20 chunks into the LLM context (noise) or only using 1 (insufficient info).
**Fix:** Start with k=5, tune based on evaluation metrics. Use re-ranking to improve signal.

### ❌ Anti-Pattern 4: No Evaluation Pipeline
**Mistake:** Deploying RAG without measuring faithfulness or relevance.
**Fix:** Build a golden dataset of Q&A pairs. Run RAGAS or TruLens regularly.

### ❌ Anti-Pattern 5: Embedding Query and Documents Differently
**Mistake:** Using one model for document embeddings and another for query embeddings.
**Fix:** **Always use the same embedding model for both queries and documents.** Vectors must be comparable.

### ❌ Anti-Pattern 6: Not Handling "No Answer" Cases
**Mistake:** LLM hallucinates when no relevant document is found.
**Fix:** Add a confidence threshold. If top similarity score < threshold, return "I don't have information on that."

---

## 9. Production Best Practices

### ✅ 1. Version Your Index
Treat your vector index like code. When you update chunking logic or embedding models, create a new index version. Keep old versions for rollback.

### ✅ 2. Implement Caching
Cache embeddings for frequent queries. Cache retrieved chunks if the index doesn't change often.

### ✅ 3. Monitor Latency and Costs
Track:
- Embedding latency (p50, p99)
- Retrieval latency
- LLM token usage and cost per query

### ✅ 4. Use Hybrid Search
Combine vector similarity with keyword matching (BM25). This handles:
- Exact matches (product IDs, error codes)
- Semantic matches (conceptual questions)

```python
# Pseudo-code: Hybrid search with Weaviate
result = client.query.get("Document", ["content"]).with_hybrid(
    query="RAG architecture",
    alpha=0.5  # 0 = pure keyword, 1 = pure vector
).with_limit(5).do()
```

### ✅ 5. Set Up Guardrails
- **Input validation:** Reject malformed or overly long queries.
- **Output validation:** Check for PII, harmful content, or off-topic responses.
- **Citation:** Always return source document IDs with responses for traceability.

### ✅ 6. Asynchronous Indexing
Don't block user queries while reindexing. Use a queue (Kafka, RabbitMQ) to process new documents in the background.

### ✅ 7. A/B Test Retrieval Strategies
Test different k values, chunking strategies, and re-rankers. Measure with real user feedback and evaluation metrics.

---

## 10. Tools & Frameworks

### Orchestration Frameworks

| Framework       | Description                                                                 | GitHub                                     |
| :-------------- | :-------------------------------------------------------------------------- | :----------------------------------------- |
| **LangChain**   | Most popular. Modular chains for RAG, agents, and tool calling.              | [github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain) |
| **LlamaIndex**  | "Data framework" for LLMs. Excellent for indexing and retrieval pipelines.  | [github.com/run-llama/llama_index](https://github.com/run-llama/llama_index) |
| **Haystack**    | End-to-end NLP framework by deepset. Strong on evaluation and deployment.    | [github.com/deepset-ai/haystack](https://github.com/deepset-ai/haystack) |
| **Semantic Kernel** | Microsoft's SDK for AI orchestration. Good for .NET and enterprise.     | [github.com/microsoft/semantic-kernel](https://github.com/microsoft/semantic-kernel) |

### LangChain RAG Example

```python
# Pseudo-code: Simple RAG with LangChain
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load and chunk
loader = TextLoader("knowledge_base.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Embed and index
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG Chain
llm = ChatOpenAI(model="gpt-4o")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Query
answer = qa_chain.invoke("What are the best practices for chunking?")
print(answer)
```

### LlamaIndex RAG Example

```python
# Pseudo-code: Simple RAG with LlamaIndex
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Load documents from a folder
documents = SimpleDirectoryReader("./data").load_data()

# Build index (embeds + stores)
index = VectorStoreIndex.from_documents(documents)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is the hybrid chunking strategy?")
print(response)
```

---

## 11. References

### Papers
- Lewis et al., 2020. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Izacard & Grave, 2021. *Leveraging Passage Retrieval with Generative Models for Open Domain QA.* [arXiv:2007.01282](https://arxiv.org/abs/2007.01282)
- Gao et al., 2023. *Retrieval-Augmented Generation for Large Language Models: A Survey.* [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)

### Leaderboards & Benchmarks
- **MTEB (Massive Text Embedding Benchmark):** [huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- **BEIR (Benchmarking IR):** [github.com/beir-cellar/beir](https://github.com/beir-cellar/beir)

### Documentation
- **LangChain Docs:** [python.langchain.com](https://python.langchain.com/)
- **LlamaIndex Docs:** [docs.llamaindex.ai](https://docs.llamaindex.ai/)
- **Pinecone Docs:** [docs.pinecone.io](https://docs.pinecone.io/)
- **RAGAS Docs:** [docs.ragas.io](https://docs.ragas.io/)

---

## 📜 License

This handbook is released under the **MIT License**.

```
MIT License

Copyright (c) 2025 The Practical RAG Systems Handbook Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  <b>🚀 Build Smarter RAG Systems. Ground Your LLMs. Ship with Confidence. 🚀</b>
</p>
