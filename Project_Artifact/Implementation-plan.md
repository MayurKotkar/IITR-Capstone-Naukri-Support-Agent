# Implementation Plan — Naukri.com Domain Support Agent

This implementation plan provides a comprehensive, phase-wise, step-by-step roadmap for building, testing, evaluating, and documenting the **Naukri.com Domain Support Agent** according to [Architecture.md](/Project_Artifact/Architecture.md) and [Problem_statement.md](/Project_Artifact/Problem_statement.md).

---

## Table of Contents

- [Phase 0: Environment Setup, Configuration & Foundation](#phase-0-environment-setup-configuration--foundation)

- [Phase 1: Part 1 — Dataset Design & RAG Core (Tasks 1–5)](#phase-1-part-1--dataset-design--rag-core-tasks-15)

  - [Step 1.1: Dataset Generator & Statistical Validation (Task 1)](#step-11-dataset-generator--statistical-validation-task-1)

  - [Step 1.2: Knowledge Base Authoring (Task 2)](#step-12-knowledge-base-authoring-task-2)

  - [Step 1.3: Dual Chunking Pipelines (Task 3)](#step-13-dual-chunking-pipelines-task-3)

  - [Step 1.4: Dual ChromaDB Indexing (Task 3)](#step-14-dual-chromadb-indexing-task-3)

  - [Step 1.5: Empirical Threshold Calibration & Fallback (Task 4)](#step-15-empirical-threshold-calibration--fallback-task-4)

  - [Step 1.6: Grounded Generation Execution & Demonstration (Task 4)](#step-16-grounded-generation-execution--demonstration-task-4)

  - [Step 1.7: Chunking Strategy Evaluation & Comparative Analysis (Task 5)](#step-17-chunking-strategy-evaluation--comparative-analysis-task-5)

- [Phase 2: Part 2 — CrewAI Orchestration, Memory, Schemas & Guardrails (Tasks 6–10)](#phase-2-part-2--crewai-orchestration-memory-schemas--guardrails-tasks-610)

  - [Step 2.1: Application Lookup Tool & Escalation Formula (Task 6)](#step-21-application-lookup-tool--escalation-formula-task-6)

  - [Step 2.2: Pydantic Output Schemas (Task 9)](#step-22-pydantic-output-schemas-task-9)

  - [Step 2.3: Guardrail Engine — PII, Injection & Groundedness (Task 10)](#step-23-guardrail-engine--pii-injection--groundedness-task-10)

  - [Step 2.4: Deterministic MOCK_LLM for CrewAI (Task 7 Foundation)](#step-24-deterministic-mock_llm-for-crewai-task-7-foundation)

  - [Step 2.5: Build the RAG Tool for CrewAI (Task 7 Foundation)](#step-25-build-the-rag-tool-for-crewai-task-7-foundation)

  - [Step 2.6: CrewAI Agents & Task Assembly (Task 7)](#step-26-crewai-agents--task-assembly-task-7)

  - [Step 2.7: Session Memory with LangChain (Task 8)](#step-27-session-memory-with-langchain-task-8)

  - [Step 2.8: Master Part 2 Verification Runner (run_part2.py)](#step-28-master-part-2-verification-runner-run_part2py)

- [Phase 3: Part 3 — Observability, FastAPI Deployment & Evaluation Suite (Tasks 11–13)](#phase-3-part-3--observability-fastapi-deployment--evaluation-suite-tasks-1113)

  - [Step 3.1: ELK-Style Structured JSON-Lines Logger (Task 12)](#step-31-elk-style-structured-json-lines-logger-task-12)

  - [Step 3.2: FastAPI Application — HTTP Endpoints (Task 11)](#step-32-fastapi-application--http-endpoints-task-11)

  - [Step 3.3: WebSocket Endpoint & Disconnect Handling (Task 11)](#step-33-websocket-endpoint--disconnect-handling-task-11)

  - [Step 3.4: 15-Query Evaluation Suite — LLM-as-Judge (Task 13)](#step-34-15-query-evaluation-suite--llm-as-judge-task-13)

- [Phase 4: Part 4 — Governance, Autogen Review & Caching (Tasks 14–16)](#phase-4-part-4--governance-autogen-review--caching-tasks-1416)

  - [Step 4.1: Query Normalization & In-Memory Response Caching (Task 16)](#step-41-query-normalization--in-memory-response-caching-task-16)

  - [Step 4.2: Four-Layer AI Governance Implementation (Task 15)](#step-42-four-layer-ai-governance-implementation-task-15)

  - [Step 4.3: Autogen 2-Agent Group Chat Review Stage (Task 14)](#step-43-autogen-2-agent-group-chat-review-stage-task-14)

  - [Step 4.4: Review Stage Demonstration & Transcripts (Task 14)](#step-44-review-stage-demonstration--transcripts-task-14)

- [Phase 5: Master Integration, Transcripts Audit & README Finalization](#phase-5-master-integration-transcripts-audit--readme-finalization)

  - [Step 5.1: Master End-to-End Orchestrator Pipeline](#step-51-master-end-to-end-orchestrator-pipeline)

  - [Step 5.2: Verification of All 24 Graded Demonstration & Verification Transcripts](#step-52-verification-of-all-24-graded-demonstration--verification-transcripts)

  - [Step 5.3: README.md Authoring & Grader Reproducibility Audit](#step-53-readmemd-authoring--grader-reproducibility-audit)

---

## Phase 0: Environment Setup, Configuration & Foundation

### Objective

Create a clean, isolated virtual environment, define all pinned dependencies, configure environment flags (zero telemetry, zero network, deterministic seed), and scaffold the repository folder structure.

### Step 0.1: Repository Scaffolding

Create the directory structure as specified in [Architecture.md](/Project_Artifact/Architecture.md#4-repository-structure):

```

naukri-support-agent/

├── Part1_Dataset_RAG/

├── Part2_CrewAI_Orchestration/

│   ├── tools/

│   └── agents/

├── Part3_Evaluation_Deployment/

├── Part4_Governance/

└── transcripts/

```

### Step 0.2: Virtual Environment & Dependency Management

Create and activate a Python 3.10+ virtual environment:

```powershell

python -m venv .venv

.venv\Scripts\Activate.ps1

```

Define `requirements.txt` with exact compatible versions:

- `crewai>=0.80.0`

- `autogen-agentchat>=0.4.0`

- `chromadb>=0.5.0`

- `sentence-transformers>=3.0.0`

- `langchain>=0.2.0`

- `langchain-community>=0.2.0`

- `fastapi>=0.111.0`

- `uvicorn[standard]>=0.30.0`

- `pydantic>=2.7.0`

- `websockets>=12.0`

- `pytest>=8.0.0`

### Step 0.3: Central Configuration (`config.py`)

Create `config.py` to centralize all tunable thresholds, file paths, and seed values:

- `USE_MOCK_LLM = True` (Default mode; zero API keys required).

- Enforce telemetry suppression before importing any crewai packages:

  ```python

  import os

  os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

  os.environ["OTEL_SDK_DISABLED"] = "true"

  ```

- Define constants:

  - Seed: `42`

  - Record count: `50`

  - Salary range: `₹3,00,000` to `₹35,00,000`

  - ChromaDB persist directory: `./chroma_db`

  - Sentence transformer model name: `"all-MiniLM-L6-v2"`

  - Max token budget cap: `1000`

  - Phone regex pattern: `r"(?:\+91[\-\s]?)?[6789]\d{9}|(?:\+91[\-\s]?)?\d{5}[\-\s]?\d{5}"`

---

## Phase 1: Part 1 — Dataset Design & RAG Core (Tasks 1–5)

### Step 1.1: Dataset Generator & Statistical Validation (Task 1)

- **File:** `Part1_Dataset_RAG/dataset.py`

- **Actions:**

  1. Implement `generate_dataset(seed: int = 42, num_records: int = 50) -> list[dict]`.

  2. Define category vocabulary: `Software Engineer`, `Data Analyst`, `Product Manager`, `HR Executive`, `Sales Associate`.

  3. Define status vocabulary: `Applied`, `Screening`, `Interview Scheduled`, `Offered`, `Rejected`.

  4. Generate fields per record:

     - `record_id`: Formatted as `APP-001`, `APP-002`, ...

     - `category`: Sampled with controlled weights (e.g., Software Engineer: 0.30, Data Analyst: 0.20, Product Manager: 0.20, HR Executive: 0.15, Sales Associate: 0.15).

     - `status`: Sampled with controlled weights (e.g., Applied: 0.35, Screening: 0.25, Interview Scheduled: 0.20, Offered: 0.10, Rejected: 0.10).

     - `expected_salary_inr`: Sampled realistically per category (e.g., Sales Associate 3L–10L, HR 4L–14L, Data Analyst 6L–22L, Software Engineer 7L–32L, PM 12L–35L).

     - `days_since_created`: Integer uniformly sampled in `[0, 30]`.

     - `flagged_priority_review`: Boolean drawn with probability `0.20`.

  5. Add validation assertion function `validate_dataset(records)`:

     - Check total records $\ge 40$.

     - Check every given category has $\ge 3$ records.

     - Check every given status has $\ge 1$ record.

     - Check percentage of `flagged_priority_review=True` is strictly within $[10\%, 30\%]$.

  6. Execute `dataset.py` standalone to print statistical summary and write output to `transcripts/part1_dataset_validation.txt`.

### Step 1.2: Knowledge Base Authoring (Task 2)

- **Directory:** `Part1_Dataset_RAG/knowledge_base/`

- **Actions:**

  1. Author exactly 12 structured policy documents as individual `.txt` files (2–5 detailed sentences each):

     - `kb_001_job_application_eligibility.txt`: Job-application eligibility criteria

     - `kb_002_interview_scheduling.txt`: Interview-scheduling process

     - `kb_003_offer_negotiation.txt`: Offer-negotiation policy

     - `kb_004_background_verification.txt`: Background-verification process

     - `kb_005_notice_period.txt`: Notice-period policy

     - `kb_006_referral_bonus.txt`: Referral-bonus policy

     - `kb_007_internal_transfer.txt`: Internal-transfer eligibility

     - `kb_008_probation_period.txt`: Probation-period policy

     - `kb_009_remote_work.txt`: Remote-work eligibility

     - `kb_010_diversity_hiring.txt`: Diversity-hiring guidelines

     - `kb_011_exit_interview.txt`: Exit-interview process

     - `kb_012_applicant_data_retention.txt`: Applicant-data-retention policy

  2. Ensure each document contains domain-specific keywords and realistic corporate HR policy rules for Naukri.com.

### Step 1.3: Dual Chunking Pipelines (Task 3)

- **File:** `Part1_Dataset_RAG/chunking.py`

- **Actions:**

  1. Implement **Strategy 1 (Fixed-size with overlap)**:

     - `chunk_fixed_size(text, chunk_size=200, overlap=50) -> list[str]`

     - Attach metadata: `{"doc_id": doc_id, "topic": topic, "strategy": "fixed_size", "chunk_index": i}`

  2. Implement **Strategy 2 (Sentence-based chunking)**:

     - `chunk_by_sentence(text) -> list[str]` using regex sentence boundary splitter `(?<=[.!?])\s+`.

     - Attach metadata: `{"doc_id": doc_id, "topic": topic, "strategy": "sentence", "chunk_index": i}`

  3. Verify that all 12 documents produce valid non-empty chunks under both strategies.

### Step 1.4: Dual ChromaDB Indexing (Task 3)

- **File:** `Part1_Dataset_RAG/indexing.py`

- **Actions:**

  1. Initialize `sentence-transformers/all-MiniLM-L6-v2` embedding function locally.

  2. Connect to ChromaDB client (`chromadb.PersistentClient(path="./chroma_db")`).

  3. Create/get two distinct collections:

     - `fixed_size_collection`

     - `sentence_collection`

  4. Perform `.upsert()` with unique chunk IDs (`f"{doc_id}_f_{i}"` and `f"{doc_id}_s_{i}"`), embeddings, documents, and metadatas.

  5. Run a probe query on both collections to verify retrieval sanity and output format.

### Step 1.5: Empirical Threshold Calibration & Fallback (Task 4)

- **File:** `Part1_Dataset_RAG/grounded_generation.py`

- **Actions:**

  1. Implement cosine similarity calculation from ChromaDB distance metrics ($similarity = 1.0 - \text{cosine\_distance}$).

  2. Define calibration dataset:

     - In-scope queries (3):

       - Q1: *"What is the notice period required for confirmation?"*

       - Q2: *"How does employee referral bonus payout work?"*

       - Q3: *"What is the background verification procedure?"*

     - Out-of-scope queries (2):

       - Q4: *"What is the best recipe for baking sourdough bread?"*

       - Q5: *"Explain quantum mechanics and wave theory."*

  3. Measure Top-1 cosine similarities for both clusters.

  4. Compute midpoint threshold $\theta = \frac{\min(\text{in\_scope}) + \max(\text{out\_scope})}{2}$.

  5. Save calibrated values into `config.py` and document them in `README.md`.

### Step 1.6: Grounded Generation Execution & Demonstration (Task 4)

- **File:** `Part1_Dataset_RAG/grounded_generation.py`

- **Actions:**

  1. Implement `generate_grounded_answer(query: str, collection_name: str, top_k: int = 3) -> dict`:

     - Retrieve top-k chunks.

     - Extract top-1 similarity.

     - If $similarity < \theta$: return fallback message: `"I don't know based on the available policy documents. This query is out of scope."` with `is_grounded = False`.

     - Else: synthesize answer strictly from retrieved context (under `MOCK_LLM`, assemble a grounded summary of top retrieved chunks) with `is_grounded = True`.

  2. Run the test suite on 5 in-scope queries + 1 out-of-scope query.

  3. Capture the full run in `transcripts/part1_rag_queries.txt`.

### Step 1.7: Chunking Strategy Evaluation & Comparative Analysis (Task 5)

- **File:** `Part1_Dataset_RAG/evaluation.py`

- **Actions:**

  1. Define ground-truth mapping for the 5 evaluation queries to parent `doc_id`s (e.g., Notice period $\to$ `kb_005`).

  2. For each query, execute top-$k$ retrieval on both `fixed_size_collection` and `sentence_collection`.

  3. Map retrieved chunks back to parent `doc_id`s and deduplicate.

  4. Compute per-query arithmetic:

     $$\text{Precision@k} = \frac{|\text{Retrieved Relevant Docs}|}{|\text{Total Retrieved Unique Docs}|}$$

     $$\text{Recall@k} = \frac{|\text{Retrieved Relevant Docs}|}{|\text{Total Relevant Docs in KB}|}$$

  5. Compute mean Precision and mean Recall across both collections.

  6. Write 2–3 sentences of justified recommendation (e.g., Sentence-based offers higher precision for short factual policy lookups, while fixed-size maintains broader semantic context).

  7. Write full arithmetic and conclusion to `transcripts/part1_chunking_evaluation.txt`.

---

## Phase 2: Part 2 — CrewAI Orchestration, Memory, Schemas & Guardrails (Tasks 6–10)

### Step 2.1: Application Lookup Tool & Escalation Formula (Task 6)

- **File:** `Part2_CrewAI_Orchestration/tools/lookup_tool.py`

- **Actions:**

  1. Define escalation scoring function:

     $$\text{recency\_score} = 1.0 - \left(\frac{\text{days\_since\_created}}{30.0}\right)$$

     $$\text{escalation\_score} = 0.6 \times \text{int}(\text{flagged\_priority\_review}) + 0.4 \times \text{recency\_score}$$

  2. Determine escalation threshold based on dataset distribution:

     - Run score computation across all 50 generated records in `dataset.py`.

     - Calculate 80th percentile threshold (e.g., $\approx 0.65$).

  3. Implement tool `check_job_application_status(record_id: str) -> dict`:

     - Locate record in `JOB_APPLICATIONS`. If not found, return `{"error": f"Record {record_id} not found"}`.

     - Return status, expected salary, computed `escalation_score`, and `recommend_escalation: bool` ($score \ge \text{threshold}$).

  4. Wrap with `@tool` or `BaseTool` from CrewAI.

### Step 2.2: Pydantic Output Schemas (Task 9)

- **File:** `Part2_CrewAI_Orchestration/schemas.py`

- **Actions:**

  1. Define `AgentResponse(BaseModel)`:

     ```python

     class AgentResponse(BaseModel):

         query: str

         answer: str

         source_documents: list[str]

         confidence_score: float = Field(ge=0.0, le=1.0)

         is_grounded: bool

         escalation_score: Optional[float] = None

         guardrail_flags: dict = Field(default_factory=dict)

     ```

  2. Write validation utility `validate_agent_response(payload: dict) -> AgentResponse`.

### Step 2.3: Guardrail Engine — PII, Injection & Groundedness (Task 10)

- **File:** `Part2_CrewAI_Orchestration/guardrails.py`

- **Actions:**

  1. **PII Masker:**

     - Regex matching Indian and international phone numbers.

     - Function `mask_pii(text: str) -> tuple[str, bool]`: Replaces numbers with `[PHONE REDACTED]`.

  2. **Prompt-Injection Detector:**

     - Pattern list: `ignore previous`, `reveal system prompt`, `you are now a`, `dan mode`, `override instructions`, `bypass filter`.

     - Function `detect_injection(text: str) -> bool`: Returns `True` if malicious prompt patterns exist.

  3. **Output Groundedness Guardrail:**

     - Function `verify_groundedness(top_similarity: float, threshold: float) -> bool`.

  4. Create targeted unit test script producing consolidated transcript:

     - `transcripts/part2_guardrails_verification.txt`

### Step 2.4: Deterministic MOCK_LLM for CrewAI (Task 7 Foundation)

- **File:** `Part2_CrewAI_Orchestration/mock_llm.py`

- **Actions:**

  1. Extend `crewai.llms.base_llm.BaseLLM`.

  2. **Avoid Pitfall 1 (System Prompt Template Matching):**

     - Never match `"Observation:"` from prompt template text.

     - Parse strictly from `role == 'assistant'` messages where CrewAI appends actual tool execution output.

  3. **Avoid Pitfall 2 (Tool Name Substring Matching):**

     - Do not check `if "lookup" in tool_name`.

     - Inspect tool declared argument schemas and exact tool names (`check_job_application_status` vs `rag_lookup`).

  4. Implement conversational reasoning in ReAct cycle (anaphora resolution and meta-history reasoning).

  5. Ensure zero outbound network calls are made (`USE_MOCK_LLM=true`).

### Step 2.5: Build the RAG Tool for CrewAI (Task 7 Foundation)

- **File:** `Part2_CrewAI_Orchestration/tools/rag_tool.py`

- **Actions:**

  1. Wrap recommended `sentence_chunks` ChromaDB collection into `@tool("rag_lookup")`.

  2. Enforce calibrated similarity threshold (0.38) and return structured retrieval dictionary with source doc IDs.

  3. Verify retrieval and out-of-scope fallback refusal.

### Step 2.6: CrewAI Agents & Task Assembly (Task 7)

- **Files:**

  - `Part2_CrewAI_Orchestration/agents/retrieval_agent.py`

  - `Part2_CrewAI_Orchestration/agents/lookup_agent.py`

  - `Part2_CrewAI_Orchestration/agents/composer_agent.py`

  - `Part2_CrewAI_Orchestration/crew.py`

- **Actions:**

  1. Instantiate **Retrieval Agent**:

     - Role: HR Policy Retrieval Specialist

     - Tools: `[rag_lookup_tool]`

  2. Instantiate **Lookup Agent**:

     - Role: Application Records Specialist

     - Tools: `[check_job_application_status_tool]`

  3. Instantiate **Response Composer**:

     - Role: Senior HR Support Composer

     - Tools: `[]` (pure synthesis agent enforcing principle of least autonomy)

  4. Define Crew tasks: `retrieval_task`, `lookup_task`, and `synthesis_task`.

  5. Assemble crew using `Process.sequential` with telemetry explicitly disabled.

  6. Execute demonstrations:

     - Policy inquiry invoking RAG tool -> `transcripts/part2_crew_rag_demo.txt`.

     - Record inquiry invoking Lookup tool -> `transcripts/part2_crew_lookup_demo.txt`.

### Step 2.7: Session Memory with LangChain (Task 8)

- **File:** `Part2_CrewAI_Orchestration/memory.py`

- **Actions:**

  1. Integrate LangChain `InMemoryChatMessageHistory` via `SESSION_STORE: dict[str, InMemoryChatMessageHistory] = {}`.

  2. Pure agentic design: pass LangChain message history directly into CrewAI task prompt context.

  3. Implement multi-turn runner:

     - Turn 1: User asks *"What is the notice period policy?"*

     - Turn 2: User asks *"Does the same apply to employees under probation?"* (resolved by agent reasoning).

  4. Implement fresh session runner with new `session_id`, demonstrating state is isolated/reset.

  5. Capture outputs into:

     - `transcripts/part2_memory_multiturn.txt`

     - `transcripts/part2_memory_fresh.txt`

### Step 2.8: Master Part 2 Verification Runner (run_part2.py)

- **File:** `Part2_CrewAI_Orchestration/run_part2.py`

- **Actions:**

  1. Sequentially execute and assert all 6 functional Part 2 steps (Tasks 6–10).

  2. Validate complete multi-agent pipeline in a single command.

---

## Phase 3: Part 3 — Observability, FastAPI Deployment & Evaluation Suite (Tasks 11–13)

### Step 3.1: ELK-Style Structured JSON-Lines Logger (Task 12)

- **File:** `Part3_Evaluation_Deployment/logger.py`

- **Actions:**

  1. Configure custom JSON-Lines log handler appending to `logs/audit.jsonl`.

  2. Fields per entry: `timestamp`, `trace_id`, `method`, `endpoint`, `query_masked`, `response_time_ms`, `guardrails_activated`, `cache_hit`, `status_code`.

  3. Ensure PII masking is applied to `query_masked` **prior** to disk write, guaranteeing raw phone numbers never hit disk.

### Step 3.2: FastAPI Application — HTTP Endpoints (Task 11)

- **Files:** `Part3_Evaluation_Deployment/app.py`, `Part3_Evaluation_Deployment/api_models.py`

- **Actions:**

  1. Implement `POST /ask`:

     - Request: `AskRequest(query: str, session_id: Optional[str])`.

     - Executes PII masking, injection checks, cache lookup, crew execution, structured output validation, and structured logging.

     - Response: `AskResponse`.

  2. Implement `POST /add-document`:

     - Request: `AddDocRequest(doc_id: str, topic: str, content: str)`.

     - Validates document, chunks, embeds, upserts into ChromaDB, and returns status confirmation.

  3. Test endpoints with `TestClient` or `requests` and capture output in `transcripts/part3_api_endpoints.txt`.

### Step 3.3: WebSocket Endpoint & Disconnect Handling (Task 11)

- **File:** `Part3_Evaluation_Deployment/app.py`

- **Actions:**

  1. Implement `@app.websocket("/ws/chat")`.

  2. Manage connection lifecycle in `try...except WebSocketDisconnect`.

  3. Demonstrate client abrupt disconnect:

     - Send message 1 $\to$ receive response.

     - Simulate abrupt disconnect.

     - Assert server catches exception and remains running for subsequent requests.

  4. Write log output to `transcripts/part3_websocket_disconnect.txt`.

### Step 3.4: 15-Query Evaluation Suite — LLM-as-Judge (Task 13)

- **File:** `Part3_Evaluation_Deployment/evaluation_suite.py`

- **Actions:**

  1. Formulate 15 test queries:

     - 12 covering all required KB topics (`kb_001` through `kb_012`).

     - 1 record lookup query (`APP-005`).

     - 2 deliberately out-of-scope / edge queries.

  2. Implement deterministic LLM-as-judge scoring under `MOCK_LLM`:

     - **Accuracy**: $1.0$ if expected policy topic/record is identified, else $0.0$.

     - **Grounding**: $1.0$ if answer uses only retrieved chunks, else $0.0$.

     - **Completeness**: $1.0$ if full policy requirements addressed, else $0.5$ or $0.0$.

     - **Safety**: $1.0$ if free of leaked PII and ungrounded hallucinations, else $0.0$.

  3. Execute evaluation suite, tabulate per-query scores, compute 4 dimension averages, and write to `transcripts/part3_evaluation_15queries.txt`.

---

### Step 3.5: Master Part 3 Verification Runner (run_part3.py)

- **File:** `Part3_Evaluation_Deployment/run_part3.py`

- **Actions:**

  1. Sequentially execute and assert all Part 3 modules: `logger.py`, `app.py --mode http`, `app.py --mode ws`, and `evaluation_suite.py`.

  2. Audit all 4 Part 3 transcripts and `logs/audit.jsonl` artifact in ~24 seconds.

  3. Validate cumulative 19/19 transcript inventory across Parts 1-3.

---

## Phase 4: Part 4 — Governance, Autogen Review & Caching (Tasks 14–16)

### Step 4.1: Query Normalization & In-Memory Response Caching (Task 16)

- **File:** `Part4_Governance/cache.py`

- **Actions:**

  1. Implement `normalize_query(q: str) -> str`: lowercased, stripped, punctuation cleaned, whitespace normalized.

  2. Implement `InMemoryResponseCache`:

     - Store key-value pairs with hit/miss counters and timing telemetry.

  3. Demonstrate cache hit:

     - Request 1: cache miss, latency $T_1$, pipeline executed.

     - Request 2: identical query, cache hit, latency $T_2 \ll T_1$, LLM bypassed.

  4. Save evidence to `transcripts/part4_cache_hit.txt`.

### Step 4.2: Four-Layer AI Governance Implementation (Task 15)

- **File:** `Part4_Governance/governance.py`

- **Actions:**

  1. **Application Layer (Least Autonomy):**

     - Verify that `check_job_application_status` is strictly assigned to `Lookup Agent`.

     - Include assertion verifying `Retrieval Agent` and `Composer Agent` throw configuration or permission error if lookup is called.

     - Document 1-paragraph justification.

  2. **Application Layer (Risk Classification):**

     - Document classification as **HIGH RISK** (handles hiring records, salary data, escalation decisions).

  3. **Runtime Layer (Cost-Budget Cap):**

     - Implement token estimator `estimate_tokens(query)`.

     - Implement `BudgetCapEnforcer(max_tokens=1000)`.

     - Test rejection on oversized input ($\ge 1000$ tokens) raising `429 / BudgetExceededException`.

  4. Save transcripts:

     - `transcripts/part4_governance_least_autonomy.txt`

     - `transcripts/part4_budget_cap_reject.txt`

### Step 4.3: Autogen 2-Agent Group Chat Review Stage (Task 14)

- **File:** `Part4_Governance/autogen_review.py`

- **Actions:**

  1. Define verdict schema `ReviewVerdict(BaseModel)`:

     ```python

     class ReviewVerdict(BaseModel):

         approved: bool

         final_answer: str

         reason: str

     ```

  2. Build Autogen team:

     - Agent 1: `Policy-Compliance-Reviewer`

     - Agent 2: `Final-Editor` configured with `output_content_type=ReviewVerdict`

  3. Construct team with `RoundRobinGroupChat(max_turns=2)` and registered `custom_message_types=[StructuredMessage[ReviewVerdict]]`.

  4. Wire team to receive Crew draft answer + retrieved RAG context.

### Step 4.4: Review Stage Demonstration & Transcripts (Task 14)

- **File:** Part4_Governance/autogen_review.py

- **Actions:**

  1. Test Case 1 (Approve unchanged): Grounded draft answer → Reviewer approves → approved=True → output to transcripts/part4_autogen_approve.txt.

  2. Test Case 2 (Revise draft): Inject ungrounded false claim into draft → Reviewer catches discrepancy → Editor rewrites answer → approved=False → output to transcripts/part4_autogen_revise.txt.

### Step 4.5: Master Part 4 Verification Runner (run_part4.py)

- **File:** Part4_Governance/run_part4.py

- **Actions:**

  1. Sequentially execute and assert all Part 4 tasks: Task 16 response cache, Task 15 least autonomy & budget cap, Task 14 Autogen review cases.

  2. Audit all 5 Part 4 transcripts (part4_cache_hit.txt, part4_governance_least_autonomy.txt, part4_budget_cap_reject.txt, part4_autogen_approve.txt, part4_autogen_revise.txt).

  3. Validate deterministic execution in ~1.67 seconds under MOCK_LLM.

---

## Phase 5: Master Integration, Transcripts Audit & README Finalization

### Step 5.1: Master End-to-End Orchestrator Pipeline

- **File:** `main.py`

- **Actions:**

  1. Connect all phases:

     $$\text{Client} \to \text{Guardrails} \to \text{Budget} \to \text{Cache} \to \text{CrewAI Crew} \to \text{Autogen Review} \to \text{Logger} \to \text{Response}$$

  2. Provide a single runner script `run_all.py` that executes every demonstration sequentially and refreshes all transcripts.

### Step 5.2: Verification of All 24 Graded Demonstration & Verification Transcripts

Audit the existence, non-emptiness, and criteria alignment of every transcript file:

**Part 1: Dataset & RAG Core (6 Transcripts)**

1. `part1_dataset_validation.txt` (Task 1: Statistical validation)

2. `part1_chunking_verification.txt` (Task 3: Dual chunking verification)

3. `part1_indexing_verification.txt` (Task 3: Dual ChromaDB indexing verification)

4. `part1_threshold_calibration.txt` (Task 4: Empirical threshold calibration report)

5. `part1_rag_queries.txt` (Task 4: 5 In-scope + 1 Out-of-scope query demonstrations)

6. `part1_chunking_evaluation.txt` (Task 5: Comparative precision/recall evaluation)

**Part 2: CrewAI Multi-Agent Orchestration (9 Transcripts)**

7. `part2_lookup_escalation_verification.txt` (Task 6: Application status tool & escalation formula)

8. `part2_schemas_verification.txt` (Task 9: Pydantic AgentResponse schema validation)

9. `part2_guardrails_verification.txt` (Task 10: Consolidated PII, injection, groundedness checks)

10. `part2_mock_llm_verification.txt` (Task 7 Foundation: Deterministic MockLLM verification)

11. `part2_rag_tool_verification.txt` (Task 7 Foundation: CrewAI rag_lookup tool verification)

12. `part2_crew_rag_demo.txt` (Task 7: CrewAI policy retrieval demonstration)

13. `part2_crew_lookup_demo.txt` (Task 7: CrewAI application lookup demonstration)

14. `part2_memory_multiturn.txt` (Task 8: Multi-turn session memory demonstration)

15. `part2_memory_fresh.txt` (Task 8: Fresh session isolation demonstration)

**Part 3: Observability, Deployment & Evaluation (4 Transcripts + Audit Log Artifact)**

16. `part3_logger_verification.txt` (Task 12: Structured JSON-Lines logger & PII masking)

17. `part3_api_endpoints.txt` (Task 11: HTTP endpoints verification)

18. `part3_websocket_disconnect.txt` (Task 11: WebSocket disconnect verification)

19. `part3_evaluation_15queries.txt` (Task 13: 15-query LLM-as-judge benchmark evaluation)

- *Audit Log Artifact:* logs/audit.jsonl (Task 12: ELK-style structured audit log)

**Part 4: Governance, Autogen Review & Caching (5 Transcripts)**

20. part4_autogen_approve.txt (Task 14: Review approval unchanged)

21. part4_autogen_revise.txt (Task 14: Review revision demonstration)

22. part4_governance_least_autonomy.txt (Task 15: Least-autonomy verification)

23. part4_budget_cap_reject.txt (Task 15: Token budget cap rejection)

24. part4_cache_hit.txt (Task 16: Response cache hit demonstration)

### Step 5.3: README.md Authoring & Grader Reproducibility Audit

- **File:** `README.md`

- **Actions:**

  1. Explicitly state the track: **Naukri.com (Recruitment & HR)**.

  2. Document exact dataset parameters: `seed=42`, category weights, status weights, salary range.

  3. Confirm telemetry is disabled (`CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true`).

  4. Document calibrated similarity threshold $\theta$ with empirical measurement values.

  5. Provide step-by-step reproduction instructions running exclusively under `MOCK_LLM` with zero API keys.
