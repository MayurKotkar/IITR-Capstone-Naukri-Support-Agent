# Phase-0

## Prompt

1. We are going to build a AI multi agentic project. read the `Project_Artifact/Problem_statement.txt` file and generate the `Problem_statement.md` file with exact same information at same location.
2. Now read `problem_statement.md` file and Generate a detailed `Architecture.md` for building this workflow.
3. Now generate a detailed phase-wise `Implementation-plan.md` using Architecture.md where we need every single step documented which we need to perform.
4. Now we need to implement phase-0. dont make any code changes. only let me know what i need to do.

## Important steps

1. Create virtual environment.
2. create folder structure for project.
3. create `requirements.txt`, `.env` and `config.py` files and install all dependencies.

<br>

# Phase-1

## Prompt

1. Just for your information phase-0 is completed. now we need to start phase-1. don't make code changes, give me the steps only for now.
2. Execute Step 1.1: Dataset Generator & Statistical Validation (Task 1). Vocabulary and constant are defined in `config.py` file. so, try to reuse instead of writing new code lines.
3. Execute Step 1.2: Author the Knowledge Base (Task 2) for all 12 documents. Instead of creating python dict or list, create 12 text files with 2-5 lines each and save in `knowledge_base` folder. Also update line 176 of `Architecture.md` as we are not creating single .py file for knowledge base.
4. Execute Step 1.3: Implement the Dual Chunking Strategies (Task 3).
5. Execute Step 1.4: Dual ChromaDB Indexing (Task 3).
6. Execute Step 1.5: Empirical Threshold Calibration & Fallback (Task 4).
7. Execute Step 1.6: Execute Grounded Generation Demo (Task 4).
8. Execute Step 1.7: Evaluate & Compare Both Chunking Strategies (Task 5).
9. Audit and verify all completed tasks in Part 1 against the capstone acceptance criteria to confirm readiness.

## Important steps

1. Implement and run `dataset.py` file. Its output is stored in `naukri-support-agent/transcripts/part1_dataset_validation.txt` file.
2. Check the `knowledge_base` folder and all 12 documents is correctly created or not.
3. Implement and run `chunking.py` file. Its output is stored in `naukri-support-agent/transcripts/part1_chunking_verification.txt` file.
4. Implement and run `indexing.py` file. Its output is stored in `naukri-support-agent/transcripts/part1_indexing_verification.txt` file. Check `chroma_db` folder where two collections are created for each chunking strategy. Run one probe query to test the retrieval.
5. Implement and run `grounded_generation.py` file (step 1.5). Its output is stored in `naukri-support-agent/transcripts/part1_threshold_calibration.txt` file and add the Empirical Threshold Calibration & Fallback details in the `README.md` file.
6. Implement and run `grounded_generation.py` file (step 1.6). Its output is stored in `naukri-support-agent/transcripts/part1_rag_queries.txt` file and verify the answer groundedness by cosine similarity for both in-scope and out-of-scope queries (No LLM use).
7. Implement and run `evaluation.py` file (step 1.7). Its output is stored in `naukri-support-agent/transcripts/part1_chunking_evaluation.txt` file and check the precision and recall values for In-Scope queries and decide the best chunking strategy for our multi-agent project.
8. Implement `run_part1.py` master runner to execute and verify all Part 1 tasks and transcripts end-to-end with a single command.
9. Verify repository cleanup: ensure `README.md` is kept exclusively at the workspace root, and all transcripts are consolidated inside `naukri-support-agent/transcripts/`.

<br>

# Phase-2

## Prompt

1. As phase-1 is completed. now we need to start phase-2. don't make code changes, give me the steps only for now.
2. Execute Step 2.1: Implement the Application Status & Escalation Tool (Task 6).
3. Execute Step 2.2: Implement Pydantic Structured Output Schemas (Task 9).
4. Execute Step 2.3: Implement Guardrails Suite (Task 10).
5. Execute Step 2.4: Implement Deterministic MOCK_LLM for CrewAI (Task 7 Foundation).
6. Execute Step 2.5: Build the RAG Tool for CrewAI (Task 7 Foundation).
7. Execute Step 2.6: Assemble the 3 CrewAI Agents & Sequential Crew (Task 7).
8. Execute Step 2.7: Implement Session Memory with LangChain (Task 8).
9. Execute Step 2.8: Build & Execute the Master Part 2 Verification Runner.

## Important steps

1. Implement and run `lookup_tool.py` file. Its output is stored in `naukri-support-agent/transcripts/part2_lookup_escalation_verification.txt` file. Application status tool is returning the record and escalation score.
2. Implement and run `schemas.py` file. Its output is stored in `naukri-support-agent/transcripts/part2_schemas_verification.txt` file. Enforces AgentResponse schema validation across RAG, lookup, and fallback responses.
3. Implement and run `guardrails.py` file. Its output is stored in `naukri-support-agent/transcripts/part2_guardrails_verification.txt` file. Consolidated verification for input PII masking, prompt-injection detection, and output groundedness refusal.
4. Implement and run `mock_llm.py` file. Its output is stored in `naukri-support-agent/transcripts/part2_mock_llm_verification.txt` file. Implements offline deterministic MockLLM subclassing BaseLLM with ReAct tool cycle and pitfall protections.
5. Implement and run `rag_tool.py` file. Its output is stored in `naukri-support-agent/transcripts/part2_rag_tool_verification.txt` file. Wraps sentence_chunks collection into @tool("rag_lookup") with 0.38 similarity threshold enforcement.
6. Implement agent factories (`retrieval_agent.py`, `lookup_agent.py`, `composer_agent.py`) and orchestrator `crew.py`. Outputs are stored in `naukri-support-agent/transcripts/part2_crew_rag_demo.txt` and `naukri-support-agent/transcripts/part2_crew_lookup_demo.txt`. Verified 3-agent sequential CrewAI execution with PII masking, injection defense, and Pydantic schema validation.
7. Implement and run `memory.py` file. Outputs are stored in `naukri-support-agent/transcripts/part2_memory_multiturn.txt` and `naukri-support-agent/transcripts/part2_memory_fresh.txt`. Demonstrates multi-turn context carryover via LangChain InMemoryChatMessageHistory and strict session isolation.
8. Implement and execute master runner `run_part2.py`. End-to-end verification passed across all 6 functional steps (lookup, schemas, guardrails, MockLLM/RAG, sequential crew, and LangChain memory).

<br>

# Phase-3

## Prompt

1. As phase-2 is completed. now we need to start phase-3. don't make code changes, give me the steps only for now.
2. Execute Step 3.1: Implement the ELK-Style Structured JSON-Lines Logger (Task 12).
3. Execute Step 3.2: Implement FastAPI HTTP Endpoints (Task 11).
4. Execute Step 3.3: Implement WebSocket Endpoint & Disconnect Handling (Task 11).
5. Execute Step 3.4: Build & Execute the 15-Query Evaluation Suite (Task 13).
6. Execute Step 3.5: Build & Execute the Master Part 3 Verification Runner.

## Important steps
  
1. Implement and test `logger.py` file. Its output is stored in `naukri-support-agent/transcripts/part3_logger_verification.txt` and structured logs in `naukri-support-agent/logs/audit.jsonl`. Guarantees ELK JSON-Lines format with trace IDs, latency, and pre-disk PII masking.
2. Implement Pydantic API schemas in `api_models.py` (`AskRequest`, `AskResponse`, `AddDocRequest`, `AddDocResponse`) and FastAPI HTTP endpoints (`POST /ask`, `POST /add-document`) in `app.py`. Its output is stored in `naukri-support-agent/transcripts/part3_api_endpoints.txt`.
3. Implement `@app.websocket("/ws/chat")` in `app.py` with `WebSocketDisconnect` exception handling ensuring the server survives abrupt client disconnection. Its output is stored in `naukri-support-agent/transcripts/part3_websocket_disconnect.txt`.
4. Implement and run `evaluation_suite.py` file (Task 13). Evaluates 15 test queries (12 KB topics, 1 lookup, 2 edge-cases) on Accuracy, Grounding, Completeness, and Safety under `MOCK_LLM`. Its output is stored in `naukri-support-agent/transcripts/part3_evaluation_15queries.txt`.
5. Implement and execute master runner `run_part3.py` to verify the entire Part 3 suite (audit logger, HTTP endpoints, WebSocket disconnect, and 15-query evaluation benchmark) in a single command.

<br>

# Phase-4

## Prompt

1. As phase-3 is completed, now we need to start phase-4. Don't make code changes, give me the steps only for now.
2. Execute Step 4.1: Implement Query Normalization & In-Memory Response Caching (Task 16).
3. Execute Step 4.2: Implement the Four-Layer AI Governance Model (Task 15).
4. Execute Step 4.3 & 4.4: Implement the Autogen 2-Agent Group Chat Review Stage (Task 14).
5. Execute Step 4.5: Build & Execute the Master Part 4 Verification Runner.

## Important steps

1. Implement and test `cache.py` file (Task 16). Implements `normalize_query()` (canonical lowercasing, punctuation stripping, whitespace collapsing) and thread-safe `InMemoryResponseCache`. Output stored in `naukri-support-agent/transcripts/part4_cache_hit.txt`, demonstrating cache hit ($T_2 \ll T_1$, 65,470x speedup, redundant LLM calls avoided).
2. Implement and test `governance.py` file (Task 15). Implements Application Layer Principle of Least Autonomy (restricting `check_job_application_status` exclusively to Lookup Agent; blocking unauthorized agents with `UnauthorizedToolAccessException`), High-Risk classification and justification, and Runtime Layer token budget cap enforcer (`MAX_TOKEN_BUDGET=1000`, rejecting oversized queries with HTTP 429 / `BudgetExceededException`). Outputs stored in `naukri-support-agent/transcripts/part4_governance_least_autonomy.txt` and `naukri-support-agent/transcripts/part4_budget_cap_reject.txt`.
3. Implement and test `autogen_review.py` file (Task 14). Constructs 2-agent Autogen Round Robin Group Chat team (`Policy_Compliance_Reviewer` + `Final_Editor`) using `RoundRobinGroupChat(max_turns=2, custom_message_types=[StructuredMessage[ReviewVerdict]])` with Pydantic `ReviewVerdict(approved: bool, final_answer: str, reason: str)`. Outputs stored in `naukri-support-agent/transcripts/part4_autogen_approve.txt` (Case 1: Grounded draft approved unchanged) and `naukri-support-agent/transcripts/part4_autogen_revise.txt` (Case 2: Injected ungrounded 10-day notice buyout claim caught and revised).
4. Implement and execute master runner `run_part4.py`. End-to-end verification passed across all 3 Phase 4 tasks (caching, governance least autonomy & budget cap, Autogen review stage).

<br>

# Phase-5

## Prompt

1. Execute Step 5.1: Build the Master End-to-End Orchestrator Pipeline (`main.py`).
2. Execute Step 5.2: Build the Master Capstone Integration & 24-Transcript Grader Audit Suite (`run_all.py`).
3. Execute Step 5.3: Update and finalize `README.md` with complete documentation, architecture diagrams, benchmark scorecards, and grader reproduction commands.

## Important steps

1. Implement `main.py` connecting the complete multi-layer pipeline: Client -> Input Guardrails (PII/Injection) -> Token Budget Enforcer -> Response Cache -> CrewAI Sequential Crew -> Autogen 2-Agent Review -> Pydantic Schema Validation -> Structured Logger -> Client. Supports `--demo`, `--query`, and `--interactive`.
2. Implement `run_all.py` executing Part 1, Part 2, Part 3, Part 4, and the End-to-End pipeline demo sequentially in ~13.25 seconds.
3. Perform comprehensive audit verifying all 24 demonstration transcripts in `naukri-support-agent/transcripts/` plus `logs/audit.jsonl` artifact (100% PASS rate).
4. Finalize `README.md` across 26 comprehensive sections documenting dataset parameters, calibration curves, governance justification, benchmark tables, and reproduction commands.
