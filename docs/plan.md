# Production-Grade RAG Chatbot — Technical Specification

## 1. Project Overview

Build a production-oriented Retrieval-Augmented Generation (RAG) chatbot.

The system will allow users to:

1. Ingest PDF documents.
2. Extract text and tables from PDFs.
3. Split documents into chunks.
4. Generate embeddings using BGE-M3.
5. Store vectors and RAG metadata in Weaviate.
6. Store application/business data in PostgreSQL.
7. Perform hybrid retrieval using:
   - Dense vector search
   - BM25 keyword search
8. Retrieve an initial Top-30 candidate set.
9. Rerank candidates using BGE-Reranker-v2-M3.
10. Select a maximum of Top-10 relevant chunks.
11. Build contextual information using parent-child retrieval.
12. Send the original user query + retrieved context to Groq.
13. Use `openai/gpt-oss-120b` as the LLM.
14. Generate an answer with citations.
15. Verify that the generated answer is grounded in retrieved context.
16. Evaluate the RAG system using a dedicated evaluation dataset.
17. Provide logging, tracing, and performance metrics.

The initial interface will be TERMINAL/CLI only.

No frontend is required for V1.

---

# 2. High-Level Architecture

```text
                         TERMINAL / CLI
                               |
                               v
                           FastAPI
                               |
                               v
                           LangGraph
                               |
                               v
                     Query Classification
                               |
                  +------------+------------+
                  |                         |
                  v                         v
            General Query               RAG Query
                  |                         |
                  v                         v
                 LLM                 Query Rewriting
                                            |
                                            v
                                  Metadata / ACL Filter
                                            |
                              +-------------+-------------+
                              |                           |
                              v                           v
                           BGE-M3                       BM25
                       Dense Retrieval             Keyword Search
                              |                           |
                              +-------------+-------------+
                                            |
                                            v
                                      Hybrid Search
                                            |
                                            v
                                         Top 30
                                            |
                                            v
                                  BGE-Reranker-v2-M3
                                            |
                                            v
                                   Relevance Threshold
                                            |
                                            v
                                   Top 10 Maximum
                                            |
                                            v
                                  Parent/Child Context
                                            |
                                            v
                                    Context Builder
                                            |
                                            v
                                         Groq API
                                            |
                                            v
                                  openai/gpt-oss-120b
                                            |
                                            v
                                  Answer + Citations
                                            |
                                            v
                                   Answer Verification
                                            |
                                            v
                                      Final Answer
````

---

# 3. Technology Stack

| Layer                | Technology                                   |
| -------------------- | -------------------------------------------- |
| Interface            | Terminal / CLI                               |
| Backend              | FastAPI                                      |
| Orchestration        | LangGraph                                    |
| LLM Provider         | Groq API                                     |
| LLM                  | `openai/gpt-oss-120b`                        |
| Application Database | PostgreSQL                                   |
| Vector Database      | Weaviate                                     |
| Embedding Model      | BGE-M3                                       |
| Keyword Retrieval    | BM25                                         |
| Retrieval            | Hybrid Search                                |
| Reranker             | BGE-Reranker-v2-M3                           |
| Document Format      | PDF                                          |
| Chunking             | Fixed-size + overlap                         |
| Initial Chunk Size   | ~600 tokens                                  |
| Initial Overlap      | ~100 tokens                                  |
| Evaluation           | Custom evaluation pipeline                   |
| Logging              | Application logging                          |
| Tracing              | RAG pipeline tracing                         |
| Metrics              | Retrieval + generation + performance metrics |

---

# 4. Database Architecture

The system uses TWO databases.

## 4.1 Weaviate

Weaviate is responsible for RAG/retrieval data.

Store:

* Chunk text
* Dense embeddings
* BM25 searchable text
* Document ID
* Document name
* Page number
* Section
* Chunk ID
* Parent ID
* Tenant ID
* Document type
* Access level
* Creation/update timestamps
* Other retrieval metadata

Example:

```json
{
  "chunk_id": "chunk_001",
  "document_id": "doc_001",
  "document_name": "employee_policy.pdf",
  "page_number": 14,
  "section": "Leave Policy",
  "parent_id": "parent_001",
  "tenant_id": "company_001",
  "document_type": "policy",
  "access_level": "internal",
  "content": "Employees are entitled to...",
  "created_at": "2026-08-25T10:00:00"
}
```

Weaviate handles:

* Vector search
* BM25
* Hybrid search
* Metadata filtering
* Vector storage
* Chunk retrieval

---

# 5. PostgreSQL

PostgreSQL is NOT used as the vector database.

PostgreSQL stores application/business data.

Recommended entities:

```text
users
documents
document_versions
chat_sessions
messages
upload_jobs
permissions
roles
tenants
audit_logs
evaluation_runs
```

Example relationship:

```text
User
 |
 +-- Chat Sessions
 |
 +-- Permissions
 |
 +-- Documents
       |
       +-- Versions
       |
       +-- Chunks in Weaviate
```

PostgreSQL should maintain the authoritative application-level relationships and permissions.

Weaviate should contain the retrieval representation of the documents/chunks.

---

# 6. Document Ingestion Pipeline

```text
PDF
 |
 v
File Validation
 |
 v
PDF Text Extraction
 |
 v
Table Extraction
 |
 v
Text Cleaning
 |
 v
Header/Footer Cleaning
 |
 v
Page Detection
 |
 v
Section Detection
 |
 v
Contextual Information
 |
 v
Fixed-size Chunking
 |
 v
600-token chunks
100-token overlap
 |
 v
Parent/Child Mapping
 |
 v
BGE-M3 Embeddings
 |
 v
Weaviate
```

---

# 7. PDF Processing

The system should support:

* Normal text PDFs
* Multi-page PDFs
* Tables
* Headers
* Footers
* Page numbers
* Sections
* Lists

OCR is NOT part of V1.

Do not introduce OCR unless explicitly requested later.

---

# 8. Chunking Strategy

Initial configuration:

```text
Chunk size: 600 tokens
Overlap: 100 tokens
```

This is a baseline and must be evaluated later.

Do NOT assume these values are optimal.

Future experiments may compare:

```text
400 / 50
600 / 100
800 / 100
1000 / 150
```

Evaluate using retrieval metrics.

---

# 9. Contextual Chunking

Each chunk should contain enough metadata/context to make it understandable independently.

Example original chunk:

```text
Employees can apply for this after 6 months.
```

Contextualized representation:

```text
Document: Employee Policy
Section: Maternity Leave
Page: 14

Employees can apply for this after 6 months.
```

The contextual representation should be embedded when appropriate.

Do NOT invent information that is not present in the source document.

---

# 10. Parent-Child Retrieval

Use a parent-child document structure.

Example:

```text
Parent Section
|
+-- Child Chunk 1
+-- Child Chunk 2
+-- Child Chunk 3
+-- Child Chunk 4
```

The smaller child chunks are optimized for retrieval.

When a child is retrieved, retrieve its parent/neighbor context when useful.

Purpose:

```text
Small chunk
    |
    v
Better retrieval precision
    |
    v
Larger parent context
    |
    v
Better LLM understanding
```

---

# 11. Embedding Model

Use:

```text
BAAI BGE-M3
```

BGE-M3 is responsible for dense embeddings.

Pipeline:

```text
Document Chunk
     |
     v
BGE-M3
     |
     v
Dense Vector
     |
     v
Weaviate
```

For queries:

```text
User Search Query
     |
     v
BGE-M3
     |
     v
Query Vector
     |
     v
Weaviate
```

---

# 12. Retrieval Strategy

Use HYBRID SEARCH.

Hybrid retrieval combines:

```text
Dense Vector Search
+
BM25 Keyword Search
```

Architecture:

```text
                     Search Query
                          |
              +-----------+-----------+
              |                       |
              v                       v
           BGE-M3                    BM25
              |                       |
              v                       v
       Vector Search            Keyword Search
              |                       |
              +-----------+-----------+
                          |
                          v
                    Hybrid Fusion
                          |
                          v
                       Top 30
```

Do not rely only on vector similarity.

BM25 is important for:

* Names
* IDs
* Product names
* Technical terms
* Exact phrases
* Numbers
* Acronyms

Dense search is important for:

* Semantic similarity
* Paraphrases
* Conceptual questions
* Natural language queries

---

# 13. Metadata Filtering

Metadata filtering must happen as part of retrieval.

Example metadata:

```text
tenant_id
document_id
document_type
department
access_level
created_at
language
```

Example:

```text
tenant_id = company_001
AND
department = HR
AND
access_level <= user's access
```

The retrieval flow should be:

```text
User
 |
 v
Authentication
 |
 v
Authorization
 |
 v
Metadata / ACL Filter
 |
 v
Hybrid Search
```

Do NOT retrieve unauthorized documents first and filter them afterward.

---

# 14. Query Classification

Before retrieval, determine whether the query requires RAG.

Example:

```text
User Query
     |
     v
Query Classifier
     |
     +----> GENERAL
     |
     +----> RAG_REQUIRED
```

Example:

```text
"What is 2 + 2?"
```

May go directly to the LLM.

Example:

```text
"What is the leave policy mentioned in employee_policy.pdf?"
```

Requires RAG.

The classifier should be implemented as a LangGraph node.

---

# 15. Query Rewriting

The system must rewrite user queries to improve retrieval quality.

IMPORTANT:

The rewritten query MUST preserve the original meaning.

The original query must always be preserved separately.

Example:

```text
Original:
"what is the leave policy for employe?"
```

Rewritten:

```text
"What is the leave policy for employees?"
```

Another example:

```text
Original:
"how pg handles many users same time?"
```

Rewritten:

```text
"How does PostgreSQL handle multiple concurrent users?"
```

---

# 16. Query Rewrite Rules

The query rewriter must:

1. Preserve intent.
2. Preserve meaning.
3. Preserve numbers.
4. Preserve dates.
5. Preserve names.
6. Preserve negation.
7. Preserve comparison direction.
8. Preserve constraints.
9. Fix spelling.
10. Fix grammar.
11. Resolve pronouns using conversation history when unambiguous.
12. Expand abbreviations only when the meaning is certain.
13. Never invent facts.
14. Never add assumptions.
15. Never answer the question.
16. Return only the improved search query.

Original query:

```text
state.original_query
```

Rewritten query:

```text
state.rewritten_query
```

Both must remain available in LangGraph state.

---

# 17. Important Query Flow

Do NOT replace the original query.

Correct:

```text
Original Query
      |
      +--------------------------+
      |                          |
      v                          v
Query Rewriter              Preserve Original
      |
      v
Rewritten Search Query
      |
      v
Retrieval
      |
      v
Context
      |
      +--------------------------+
                                 |
                                 v
                                LLM
                                 |
                     Original Query + Context
                                 |
                                 v
                              Answer
```

The rewritten query is primarily for retrieval.

The original user question is used to formulate the final answer.

---

# 18. Retrieval Candidate Strategy

Do NOT retrieve only Top 10 initially.

Use two stages.

## Stage 1

Hybrid search:

```text
All documents
     |
     v
Metadata filtering
     |
     v
Hybrid Search
     |
     v
Top 30
```

Goal:

```text
High Recall
```

## Stage 2

Reranking:

```text
Top 30
   |
   v
BGE-Reranker-v2-M3
   |
   v
Relevance scores
   |
   v
Sort
   |
   v
Relevance threshold
   |
   v
Top 10 maximum
```

The final number may be:

```text
3
5
7
10
```

depending on relevance.

Do NOT force exactly 10 chunks.

---

# 19. Reranker

Use:

```text
BAAI BGE-Reranker-v2-M3
```

Purpose:

Improve ranking of retrieved chunks.

Pipeline:

```text
Top 30 candidates
       |
       v
BGE-Reranker-v2-M3
       |
       v
Relevance scores
       |
       v
Rank
       |
       v
Top 10 maximum
```

---

# 20. Context Construction

Before sending context to the LLM:

```text
Reranked chunks
      |
      v
Remove duplicates
      |
      v
Retrieve parent/neighbor context
      |
      v
Organize by document/page
      |
      v
Build structured context
```

Each context item should preserve source metadata:

```text
Document name
Page number
Section
Chunk ID
Source ID
```

---

# 21. Context Compression

NOT part of V1.

Keep it as a future enhancement.

Future pipeline:

```text
Top 10
  |
  v
Context Compression
  |
  v
Only relevant passages
  |
  v
LLM
```

Do not implement this initially.

---

# 22. Multi-Query Retrieval

NOT part of V1.

Keep it for difficult/complex queries later.

Future:

```text
Original Query
     |
     v
Generate multiple search queries
     |
     +---- Query 1
     +---- Query 2
     +---- Query 3
     |
     v
Multiple Retrievals
     |
     v
Fusion
     |
     v
Reranking
```

Do not implement initially.

---

# 23. LLM

Use Groq API.

Model:

```text
openai/gpt-oss-120b
```

The LLM is responsible for:

* Understanding retrieved context
* Reasoning
* Generating the final answer
* Following answer/citation instructions

The LLM must NOT be allowed to invent information when the answer is not supported by retrieved context.

---

# 24. Grounded Answer Policy

The final generation prompt should enforce:

```text
Answer using the supplied context.

Do not invent information.

If the context does not contain enough information,
clearly state that the information could not be found.

Do not use external knowledge unless explicitly allowed.

Preserve important numbers, dates, names, and conditions.

Provide citations to the retrieved source.
```

---

# 25. Citations

Every retrieved chunk should preserve:

```text
document_id
document_name
page_number
section
chunk_id
```

Example final response:

```text
Employees are entitled to 20 days of annual leave.

Source:
employee_policy.pdf — Page 14
```

Citations should be generated from actual retrieved metadata.

Do NOT hallucinate page numbers.

---

# 26. Answer Verification

After generation:

```text
Context
   |
   v
LLM
   |
   v
Draft Answer
   |
   v
Verification
   |
   +---- Supported ----> Final Answer
   |
   +---- Unsupported --> Regenerate / Refuse
```

The verification step should check:

* Is the answer supported?
* Are important claims grounded?
* Are citations correct?
* Did the model introduce unsupported facts?

---

# 27. Document Deduplication

Before ingestion:

```text
PDF
 |
 v
Hash
 |
 v
Check PostgreSQL
 |
 +---- Existing ---> Skip / Version Check
 |
 +---- New -------> Process
```

This prevents duplicate ingestion.

---

# 28. Document Versioning

Documents can have versions.

Example:

```text
employee_policy
|
+-- v1
|
+-- v2
|
+-- v3 ACTIVE
```

Metadata:

```text
document_id
version
is_active
created_at
updated_at
```

Retrieval should normally use the active version unless explicitly requested otherwise.

---

# 29. Authentication / Authorization

Production architecture should eventually support:

```text
User
 |
 v
Authentication
 |
 v
Authorization
 |
 v
Allowed Documents
 |
 v
Metadata Filter
 |
 v
Retrieval
```

Do not rely only on the LLM to enforce permissions.

Authorization must happen before retrieval.

---

# 30. Tenant Isolation

For multi-tenant systems:

```text
tenant_id
```

must be included in document metadata.

Example:

```text
Tenant A
 |
 +-- Documents

Tenant B
 |
 +-- Documents
```

A query from Tenant A must never retrieve Tenant B data.

---

# 31. Rate Limiting

Apply rate limiting to:

```text
PDF upload
Query API
LLM requests
Embedding requests
```

Redis is NOT being used for V1.

Use a simple application/API-level rate-limiting implementation initially and introduce Redis later only if distributed rate limiting becomes necessary.

---

# 32. Background Ingestion

PDF processing should be asynchronous for production.

```text
Upload PDF
    |
    v
Create Upload Job
    |
    v
Return Job ID
    |
    v
Background Worker
    |
    +-- Extract
    +-- Clean
    +-- Chunk
    +-- Embed
    +-- Store
    |
    v
Update Job Status
```

---

# 33. Logging

Log important information:

```text
request_id
user_id
query
rewritten_query
document_ids
retrieved_chunks
retrieval_scores
reranker_scores
LLM model
latency
tokens
errors
citations
```

Avoid logging sensitive document content unnecessarily.

---

# 34. Observability

Track the full pipeline:

```text
Request
 |
 v
LangGraph
 |
 +-- Query Classification
 |
 +-- Query Rewrite
 |
 +-- Embedding
 |
 +-- Weaviate
 |
 +-- BM25
 |
 +-- Hybrid Search
 |
 +-- Reranker
 |
 +-- Context Builder
 |
 +-- Groq
 |
 +-- Verification
 |
 v
Response
```

Track latency for each step.

---

# 35. Evaluation System

Evaluation must be part of the project from the beginning.

Create a golden evaluation dataset.

Example:

```json
{
  "question": "What is the employee leave entitlement?",
  "expected_answer": "...",
  "expected_document": "employee_policy.pdf",
  "expected_page": 14
}
```

Initial dataset:

```text
50–100 questions
```

Include:

* Simple questions
* Complex questions
* Exact keyword questions
* Semantic questions
* Follow-up questions
* Ambiguous questions
* Questions with no answer
* Questions requiring multiple chunks

---

# 36. Retrieval Evaluation

Measure:

| Metric      | Purpose                                  |
| ----------- | ---------------------------------------- |
| Recall@5    | Correct chunk in top 5                   |
| Recall@10   | Correct chunk in top 10                  |
| Recall@30   | Correct chunk found by initial retrieval |
| MRR         | Position of correct result               |
| NDCG@10     | Ranking quality                          |
| Precision@K | Relevance of retrieved results           |

Important:

```text
Hybrid Search → Evaluate Recall@30

Reranker → Evaluate MRR / NDCG@10
```

---

# 37. Generation Evaluation

Measure:

| Metric             | Purpose                           |
| ------------------ | --------------------------------- |
| Faithfulness       | Answer supported by context       |
| Answer Relevance   | Answers user's question           |
| Context Relevance  | Retrieved context is useful       |
| Citation Accuracy  | Citation points to correct source |
| Completeness       | Important information included    |
| Hallucination Rate | Unsupported claims                |

---

# 38. Performance Evaluation

Track:

```text
query_rewrite_latency
embedding_latency
weaviate_latency
hybrid_search_latency
reranker_latency
context_build_latency
llm_latency
total_latency
```

Also track:

```text
input_tokens
output_tokens
total_tokens
error_rate
throughput
```

---

# 39. RAG Experiments

The system should support controlled experiments.

## Chunking

```text
400 / 50
600 / 100
800 / 100
1000 / 150
```

## Retrieval

```text
Dense only
BM25 only
Hybrid
```

## Reranking

```text
No reranker
BGE-Reranker-v2-M3
```

## Candidate count

```text
Top 10
Top 20
Top 30
Top 50
```

Compare:

```text
Recall
MRR
NDCG
Faithfulness
Answer quality
Latency
Token usage
```

---

# 40. V1 Features

Implement these first:

```text
[YES]

PDF ingestion
Text extraction
Table extraction
Text cleaning
Fixed-size chunking
Chunk overlap
Contextual chunking
Parent-child structure
BGE-M3
Weaviate
BM25
Hybrid search
Metadata filtering
Top 30 retrieval
BGE-Reranker-v2-M3
Top 10 maximum
Relevance threshold
Query classification
Meaning-preserving query rewriting
Conversation history
PostgreSQL
Groq API
openai/gpt-oss-120b
Context construction
Citations
Answer verification
Document deduplication
Document versioning
Logging
Evaluation
Metrics
```

---

# 41. Deferred Features

Do NOT implement these in V1:

```text
[DEFERRED]

Multi-query retrieval
Context compression
Semantic caching
Graph RAG
Multi-vector / ColBERT
Advanced agentic retrieval
Adaptive retrieval
Self-RAG
Corrective RAG
Next.js frontend
Redis
OCR
```

These can be added after the baseline system has been evaluated.

---

# 42. Final V1 Pipeline

```text
                    TERMINAL
                       |
                       v
                    FastAPI
                       |
                       v
                   LangGraph
                       |
                       v
              Query Classification
                       |
             +---------+---------+
             |                   |
             v                   v
          GENERAL              RAG
             |                   |
             v                   v
            LLM            Query Rewrite
                                 |
                                 v
                       Metadata / ACL Filter
                                 |
                       +---------+---------+
                       |                   |
                       v                   v
                    BGE-M3                BM25
                       |                   |
                       v                   v
                 Dense Search        Keyword Search
                       |                   |
                       +---------+---------+
                                 |
                                 v
                           Hybrid Search
                                 |
                                 v
                              Top 30
                                 |
                                 v
                     BGE-Reranker-v2-M3
                                 |
                                 v
                       Relevance Threshold
                                 |
                                 v
                         Top 10 Maximum
                                 |
                                 v
                      Parent/Child Context
                                 |
                                 v
                        Context Builder
                                 |
                                 v
                            Groq API
                                 |
                                 v
                      openai/gpt-oss-120b
                                 |
                                 v
                       Answer + Citations
                                 |
                                 v
                       Answer Verification
                                 |
                                 v
                         Final Response
```

---

# 43. Final Technology Summary

The core stack is:

```text
Interface
    → Terminal / CLI

Backend
    → FastAPI

Orchestration
    → LangGraph

LLM
    → Groq API
    → openai/gpt-oss-120b

Application Database
    → PostgreSQL

Vector Database
    → Weaviate

Embedding
    → BGE-M3

Keyword Retrieval
    → BM25

Retrieval
    → Hybrid Search

Reranking
    → BGE-Reranker-v2-M3

Chunking
    → Fixed-size + 100-token overlap
    → ~600-token chunks

Advanced Retrieval
    → Contextual Chunking
    → Parent-Child Retrieval
    → Query Classification
    → Meaning-Preserving Query Rewriting

Generation
    → Grounded Answer
    → Citations
    → Answer Verification

Production
    → Authentication
    → Authorization/RBAC
    → Tenant Isolation
    → Rate Limiting
    → Background Ingestion
    → Logging
    → Tracing
    → Metrics

Evaluation
    → Recall@K
    → MRR
    → NDCG
    → Faithfulness
    → Answer Relevance
    → Citation Accuracy
    → Latency
    → Token Usage

Deferred
    → Multi-query Retrieval
    → Context Compression
    → Semantic Cache
    → Graph RAG
    → ColBERT/Multi-vector
    → Agentic Retrieval
    → Redis
    → Next.js
    → OCR
```

**Important implementation principle:** build the system as a **baseline RAG first**, measure it with the evaluation dataset, and then introduce advanced techniques one at a time. This lets you prove whether each addition actually improves retrieval quality instead of creating a complex RAG pipeline that you cannot diagnose.
