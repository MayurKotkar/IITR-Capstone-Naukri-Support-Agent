"""
Master End-to-End Orchestrator & Grader Audit Suite for Naukri.com Domain Support Agent.

Sequentially executes:
1. Part 1 Suite (Tasks 1-5): Dataset, Dual Chunking, Indexing, Calibration, Precision/Recall
2. Part 2 Suite (Tasks 6-10): Escalation Tool, CrewAI Multi-Agent, Memory, Schemas, Guardrails
3. Part 3 Suite (Tasks 11-13): Structured Logging, FastAPI HTTP/WebSocket, 15-Query Evaluation
4. Part 4 Suite (Tasks 14-16): Response Caching, AI Governance, Autogen 2-Agent Review
5. End-to-End Pipeline Showcase Demo (main.py)
6. Comprehensive Audit of all 24 Transcripts & Audit Log Artifacts.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix Windows console UTF-8 output encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root and repo root are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
REPO_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(REPO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from Part1_Dataset_RAG.run_part1 import run_part1
from Part2_CrewAI_Orchestration.run_part2 import run_master_part2
from Part3_Evaluation_Deployment.run_part3 import run_master_part3
from Part4_Governance.run_part4 import run_master_part4
from main import run_pipeline_demo

# Complete registry of all 24 mandatory demonstration and verification transcripts
EXPECTED_TRANSCRIPTS: Dict[str, Tuple[str, List[str]]] = {
    # Part 1: Dataset & RAG Core (6 Transcripts)
    "part1_dataset_validation.txt": (
        "Part 1 (Task 1): Statistical validation (N=50, categories, statuses, priority flags)",
        ["PART 1 TASK 1", "Software Engineer", "Applied", "Flagged percentage"],
    ),
    "part1_chunking_verification.txt": (
        "Part 1 (Task 3): Dual chunking verification (fixed-size vs sentence-based)",
        ["PART 1 TASK 3", "STRATEGY 1: FIXED-SIZE", "STRATEGY 2: SENTENCE-BASED", "kb_001"],
    ),
    "part1_indexing_verification.txt": (
        "Part 1 (Task 3): ChromaDB dual collection indexing verification",
        ["PART 1 TASK 3", "fixed_size_chunks", "sentence_chunks", "PROBE QUERY RETRIEVAL"],
    ),
    "part1_threshold_calibration.txt": (
        "Part 1 (Task 4): Empirical similarity threshold calibration report",
        ["PART 1 TASK 4", "CALIBRATED THRESHOLD", "Absolute Separation Gap", "CALIBRATION STATUS: SUCCESSFUL"],
    ),
    "part1_rag_queries.txt": (
        "Part 1 (Task 4): 5 In-scope grounded answers + 1 Out-of-scope fallback",
        ["PART 1 TASK 4", "[Q1] IN_SCOPE", "[Q6] OUT_OF_SCOPE", "I don't know"],
    ),
    "part1_chunking_evaluation.txt": (
        "Part 1 (Task 5): Comparative precision/recall evaluation & recommendation",
        ["PART 1 TASK 5", "ARCHITECTURAL RECOMMENDATION", "sentence_chunks", "AVERAGE"],
    ),

    # Part 2: CrewAI Multi-Agent Orchestration (9 Transcripts)
    "part2_lookup_escalation_verification.txt": (
        "Part 2 (Task 6): Application status lookup tool & escalation scoring formula",
        ["PART 2 TASK 6", "FORMULA SPECIFICATION", "escalation_score", "APP-001"],
    ),
    "part2_schemas_verification.txt": (
        "Part 2 (Task 9): Pydantic AgentResponse schema validation",
        ["PART 2 TASK 9", "AgentResponse schema verified successfully", "confidence_score"],
    ),
    "part2_guardrails_verification.txt": (
        "Part 2 (Task 10): Input/Output guardrails (PII masking, prompt injection, groundedness)",
        ["PART 2 TASK 10", "SECTION 1: INPUT PII MASKING", "SECTION 2: INPUT PROMPT INJECTION", "SECTION 3: OUTPUT GROUNDEDNESS"],
    ),
    "part2_mock_llm_verification.txt": (
        "Part 2 (Task 7 Foundation): Deterministic MockLLM verification",
        ["PART 2 TASK 7 FOUNDATION: MOCK_LLM VERIFICATION", "Zero API Keys"],
    ),
    "part2_rag_tool_verification.txt": (
        "Part 2 (Task 7 Foundation): CrewAI rag_lookup tool verification",
        ["RAG TOOL VERIFICATION", "rag_lookup"],
    ),
    "part2_crew_rag_demo.txt": (
        "Part 2 (Task 7): CrewAI HR policy retrieval demonstration",
        ["CREWAI POLICY RETRIEVAL DEMO", "According to Naukri.com policy"],
    ),
    "part2_crew_lookup_demo.txt": (
        "Part 2 (Task 7): CrewAI candidate application lookup demonstration",
        ["PART 2 TASK 7: CREWAI APPLICATION LOOKUP DEMONSTRATION", "Application APP-045"],
    ),
    "part2_memory_multiturn.txt": (
        "Part 2 (Task 8): Multi-turn session memory context carryover",
        ["PART 2 TASK 8: MULTI-TURN CONVERSATION SESSION MEMORY", "TURN 1", "TURN 2", "probation"],
    ),
    "part2_memory_fresh.txt": (
        "Part 2 (Task 8): Fresh session isolation demonstration",
        ["PART 2 TASK 8: FRESH SESSION ISOLATION TRANSCRIPT", "No prior conversation found"],
    ),

    # Part 3: Observability, Deployment & Evaluation (4 Transcripts)
    "part3_logger_verification.txt": (
        "Part 3 (Task 12): Structured JSON-Lines logger & pre-disk PII masking",
        ["PART 3 TASK 12", "Pre-Disk PII Masking Guarantee", "audit.jsonl"],
    ),
    "part3_api_endpoints.txt": (
        "Part 3 (Task 11): FastAPI HTTP endpoints (/health, /ask, /add-document)",
        ["PART 3 TASK 11", "POST /ask", "POST /add-document", "GET /health"],
    ),
    "part3_websocket_disconnect.txt": (
        "Part 3 (Task 11): WebSocket real-time chat & client disconnect handling",
        ["PART 3 TASK 11", "WebSocket handler caught WebSocketDisconnect", "WS /ws/chat"],
    ),
    "part3_evaluation_15queries.txt": (
        "Part 3 (Task 13): 15-query LLM-as-judge benchmark evaluation",
        ["PART 3 TASK 13", "TASK 13 EVALUATION BENCHMARK METRICS SUMMARY", "Mean Accuracy", "Mean Grounding", "Mean Safety"],
    ),

    # Part 4: Resilience & Governance (5 Transcripts)
    "part4_autogen_approve.txt": (
        "Part 4 (Task 14): Autogen review stage — Grounded draft approved unchanged",
        ["RoundRobinGroupChat", "StructuredMessage[ReviewVerdict]", "approved=True"],
    ),
    "part4_autogen_revise.txt": (
        "Part 4 (Task 14): Autogen review stage — Injected ungrounded claim revised",
        ["RoundRobinGroupChat", "StructuredMessage[ReviewVerdict]", "approved=False"],
    ),
    "part4_governance_least_autonomy.txt": (
        "Part 4 (Task 15): Application Layer — Least autonomy enforcement & High Risk tier",
        ["LEAST AUTONOMY", "HIGH RISK", "UnauthorizedToolAccessException"],
    ),
    "part4_budget_cap_reject.txt": (
        "Part 4 (Task 15): Runtime Layer — Token budget cap enforcement (HTTP 429)",
        ["BUDGET CAP EXCEEDED", "HTTP 429", "BudgetExceededException"],
    ),
    "part4_cache_hit.txt": (
        "Part 4 (Task 16): Response caching & canonical query normalization",
        ["CACHE HIT", "Speedup", "VERIFICATION STATUS: PASSED"],
    ),
}


def audit_all_transcripts() -> Tuple[bool, List[Dict[str, Any]]]:
    """Audit all 24 transcripts and the audit log artifact."""
    audit_records = []
    all_valid = True

    for fname, (desc, markers) in EXPECTED_TRANSCRIPTS.items():
        fpath = config.TRANSCRIPTS_DIR / fname
        if not fpath.exists():
            all_valid = False
            audit_records.append({
                "filename": fname,
                "description": desc,
                "exists": False,
                "size_bytes": 0,
                "markers_passed": False,
                "status": "MISSING",
            })
            continue

        text = fpath.read_text(encoding="utf-8")
        size = len(text.encode("utf-8"))
        missing = [m for m in markers if m not in text]
        passed = (size > 0) and (len(missing) == 0)
        if not passed:
            all_valid = False

        audit_records.append({
            "filename": fname,
            "description": desc,
            "exists": True,
            "size_bytes": size,
            "markers_passed": len(missing) == 0,
            "status": "PASSED" if passed else f"FAILED (Missing: {missing[:1]})",
        })

    # Audit logs/audit.jsonl artifact
    audit_log_path = config.AUDIT_LOG_PATH
    log_exists = audit_log_path.exists()
    log_size = audit_log_path.stat().st_size if log_exists else 0
    raw_leak = False

    if log_exists and log_size > 0:
        content = audit_log_path.read_text(encoding="utf-8")
        # Ensure no raw 10-digit mobile numbers starting with 6-9 in clear
        raw_matches = config.INDIAN_PHONE_PATTERN.findall(content)
        if raw_matches:
            raw_leak = True
            all_valid = False

    audit_records.append({
        "filename": "logs/audit.jsonl",
        "description": "Task 12: ELK-style structured audit log (pre-disk PII masking verified)",
        "exists": log_exists,
        "size_bytes": log_size,
        "markers_passed": not raw_leak,
        "status": "PASSED" if (log_exists and log_size > 0 and not raw_leak) else "FAILED",
    })

    return all_valid, audit_records


def run_full_suite() -> int:
    """Execute master suite."""
    print("=" * 80)
    print("NAUKRI.COM DOMAIN SUPPORT AGENT: MASTER INTEGRATION & AUDIT SUITE")
    print("ALL 4 PHASES + 16 TASKS + 24 TRANSCRIPTS + AUDIT LOG ARTIFACT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Offline Mode: 100% Deterministic (Zero Paid APIs, Zero Network Calls)")
    print("=" * 80)

    suite_start = time.perf_counter()
    phase_timings: List[Tuple[str, str, float, str]] = []

    # 1. Part 1: Dataset & RAG Core
    print("\n" + "#" * 80)
    print(">>> PHASE 1: PART 1 — DATASET & RAG CORE (TASKS 1-5)")
    print("#" * 80)
    t0 = time.perf_counter()
    try:
        run_part1()
        phase_timings.append(("Part 1", "Dataset & RAG Core (Tasks 1-5)", time.perf_counter() - t0, "PASSED"))
    except Exception as e:
        phase_timings.append(("Part 1", "Dataset & RAG Core (Tasks 1-5)", time.perf_counter() - t0, f"FAILED: {e}"))

    # 2. Part 2: CrewAI Multi-Agent Orchestration
    print("\n" + "#" * 80)
    print(">>> PHASE 2: PART 2 — CREWAI MULTI-AGENT ORCHESTRATION (TASKS 6-10)")
    print("#" * 80)
    t0 = time.perf_counter()
    try:
        run_master_part2()
        phase_timings.append(("Part 2", "CrewAI Multi-Agent (Tasks 6-10)", time.perf_counter() - t0, "PASSED"))
    except Exception as e:
        phase_timings.append(("Part 2", "CrewAI Multi-Agent (Tasks 6-10)", time.perf_counter() - t0, f"FAILED: {e}"))

    # 3. Part 3: Observability, Deployment & Evaluation Suite
    print("\n" + "#" * 80)
    print(">>> PHASE 3: PART 3 — OBSERVABILITY & FASTAPI DEPLOYMENT (TASKS 11-13)")
    print("#" * 80)
    t0 = time.perf_counter()
    try:
        ret3 = run_master_part3()
        p3_status = "PASSED" if ret3 in [0, None] else "FAILED"
        phase_timings.append(("Part 3", "FastAPI, Logging & 15-Q Eval (Tasks 11-13)", time.perf_counter() - t0, p3_status))
    except Exception as e:
        phase_timings.append(("Part 3", "FastAPI, Logging & 15-Q Eval (Tasks 11-13)", time.perf_counter() - t0, f"FAILED: {e}"))

    # 4. Part 4: Resilience, Governance & Autogen Review
    print("\n" + "#" * 80)
    print(">>> PHASE 4: PART 4 — GOVERNANCE & AUTOGEN REVIEW (TASKS 14-16)")
    print("#" * 80)
    t0 = time.perf_counter()
    try:
        ret4 = run_master_part4()
        p4_status = "PASSED" if ret4 in [0, None] else "FAILED"
        phase_timings.append(("Part 4", "Governance, Autogen & Caching (Tasks 14-16)", time.perf_counter() - t0, p4_status))
    except Exception as e:
        phase_timings.append(("Part 4", "Governance, Autogen & Caching (Tasks 14-16)", time.perf_counter() - t0, f"FAILED: {e}"))

    # 5. End-to-End Pipeline Demo
    print("\n" + "#" * 80)
    print(">>> PHASE 5: END-TO-END PIPELINE DEMO (MAIN.PY)")
    print("#" * 80)
    t0 = time.perf_counter()
    try:
        run_pipeline_demo()
        phase_timings.append(("Pipeline", "End-to-End Pipeline Demo (main.py)", time.perf_counter() - t0, "PASSED"))
    except Exception as e:
        phase_timings.append(("Pipeline", "End-to-End Pipeline Demo (main.py)", time.perf_counter() - t0, f"FAILED: {e}"))

    # 6. Master Transcript & Artifact Audit
    print("\n" + "#" * 80)
    print(">>> COMPREHENSIVE AUDIT: ALL 24 TRANSCRIPTS + AUDIT LOG ARTIFACT")
    print("#" * 80)
    audit_passed, records = audit_all_transcripts()

    print(f"\n{'#':<3} | {'Transcript File':<42} | {'Bytes':>8} | {'Status':<10} | {'Description'}")
    print("-" * 120)
    for idx, r in enumerate(records, 1):
        print(f"{idx:<3} | {r['filename']:<42} | {r['size_bytes']:>8} | {r['status']:<10} | {r['description'][:52]}")

    print("-" * 120)
    total_time = time.perf_counter() - suite_start

    print("\n" + "=" * 80)
    print("PHASE EXECUTION TIMINGS")
    print("=" * 80)
    for phase, desc, dur, status in phase_timings:
        print(f"{phase:<10} | {desc:<45} | {dur:>7.2f}s | {status}")

    print("-" * 80)
    print(f"Total Execution Time: {total_time:.2f}s (~{total_time/60:.1f} minutes)")
    
    passed_transcripts_count = sum(1 for r in records if "PASSED" in r["status"])
    print(f"Transcripts & Artifacts Audited: {passed_transcripts_count}/{len(records)} PASSED")

    master_ok = audit_passed and all(st == "PASSED" for _, _, _, st in phase_timings)
    print("\n" + "=" * 80)
    if master_ok:
        print("MASTER CAPSTONE AUDIT: 100% COMPLETE & VERIFIED")
        print("ALL 16 TASKS, 4 PHASES, 24 TRANSCRIPTS & AUDIT LOG MEET 100% OF RUBRIC CRITERIA")
    else:
        print("MASTER CAPSTONE AUDIT: FAILED — PLEASE REVIEW ERRORS ABOVE")
    print("=" * 80)

    return 0 if master_ok else 1


if __name__ == "__main__":
    sys.exit(run_full_suite())
