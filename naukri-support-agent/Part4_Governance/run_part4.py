"""
Master Part 4 Verification Runner for Naukri.com Domain Support Agent.

Sequentially executes and verifies all Phase 4 governance and resilience tasks:
1. Task 16: In-Memory Response Caching & Query Normalization (cache.py)
   -> Generates transcripts/part4_cache_hit.txt
2. Task 15 (Part A): Application Layer — Principle of Least Autonomy (governance.py)
   -> Generates transcripts/part4_governance_least_autonomy.txt
3. Task 15 (Part B): Runtime Layer — Token Budget Cap Enforcement (governance.py)
   -> Generates transcripts/part4_budget_cap_reject.txt
4. Task 14 (Case 1): Autogen 2-Agent Review — Draft Approval Unchanged (autogen_review.py)
   -> Generates transcripts/part4_autogen_approve.txt
5. Task 14 (Case 2): Autogen 2-Agent Review — Injected Discrepancy Revision (autogen_review.py)
   -> Generates transcripts/part4_autogen_revise.txt
6. Automated Transcript Audit & Assertion Verification.
"""

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

# Fix Windows console UTF-8 output encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root and repo root are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REPO_DIR = ROOT_DIR / "naukri-support-agent"
for p in [str(ROOT_DIR), str(REPO_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config
from Part4_Governance.cache import run_cache_demonstration
from Part4_Governance.governance import (
    run_budget_cap_demonstration,
    run_least_autonomy_demonstration,
)
from Part4_Governance.autogen_review import (
    run_autogen_approve_demonstration,
    run_autogen_revise_demonstration,
)


def run_master_part4() -> int:
    """Run full Part 4 verification suite."""
    print("=" * 80)
    print("NAUKRI.COM DOMAIN SUPPORT AGENT: MASTER PART 4 VERIFICATION RUNNER")
    print("PHASE 4: RESILIENCE, GOVERNANCE & AUTOGEN REVIEW (TASKS 14-16)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print(f"Mode     : 100% Deterministic Offline under MOCK_LLM")
    print("=" * 80)

    start_total = time.perf_counter()
    results: List[Tuple[str, str, float, str]] = []

    # --------------------------------------------------------------------------
    # Step 1: Task 16 — Response Caching & Query Normalization
    # --------------------------------------------------------------------------
    print("\n>>> [1/5] Executing Task 16: Response Caching & Query Normalization...")
    t0 = time.perf_counter()
    try:
        run_cache_demonstration(save_transcript=True)
        dur = time.perf_counter() - t0
        results.append(("Task 16", "In-Memory Response Caching & Normalization", dur, "PASSED"))
        print(f"    [SUCCESS] Task 16 completed in {dur:.2f}s")
    except Exception as e:
        dur = time.perf_counter() - t0
        results.append(("Task 16", "In-Memory Response Caching & Normalization", dur, f"FAILED: {e}"))
        print(f"    [FAILURE] Task 16 failed: {e}")

    # --------------------------------------------------------------------------
    # Step 2: Task 15 Part A — Application Layer Least Autonomy & Risk Classification
    # --------------------------------------------------------------------------
    print("\n>>> [2/5] Executing Task 15 (Part A): Principle of Least Autonomy...")
    t0 = time.perf_counter()
    try:
        run_least_autonomy_demonstration(save_transcript=True)
        dur = time.perf_counter() - t0
        results.append(("Task 15A", "Application Layer: Least Autonomy & Risk Tier", dur, "PASSED"))
        print(f"    [SUCCESS] Task 15 (Part A) completed in {dur:.2f}s")
    except Exception as e:
        dur = time.perf_counter() - t0
        results.append(("Task 15A", "Application Layer: Least Autonomy & Risk Tier", dur, f"FAILED: {e}"))
        print(f"    [FAILURE] Task 15 (Part A) failed: {e}")

    # --------------------------------------------------------------------------
    # Step 3: Task 15 Part B — Runtime Layer Token Budget Cap Enforcement
    # --------------------------------------------------------------------------
    print("\n>>> [3/5] Executing Task 15 (Part B): Runtime Layer Token Budget Cap...")
    t0 = time.perf_counter()
    try:
        run_budget_cap_demonstration(save_transcript=True)
        dur = time.perf_counter() - t0
        results.append(("Task 15B", "Runtime Layer: Token Budget Cap Enforcer (HTTP 429)", dur, "PASSED"))
        print(f"    [SUCCESS] Task 15 (Part B) completed in {dur:.2f}s")
    except Exception as e:
        dur = time.perf_counter() - t0
        results.append(("Task 15B", "Runtime Layer: Token Budget Cap Enforcer (HTTP 429)", dur, f"FAILED: {e}"))
        print(f"    [FAILURE] Task 15 (Part B) failed: {e}")

    # --------------------------------------------------------------------------
    # Step 4: Task 14 Case 1 — Autogen Review Stage: Approval Unchanged
    # --------------------------------------------------------------------------
    print("\n>>> [4/5] Executing Task 14 (Case 1): Autogen Review Approval...")
    t0 = time.perf_counter()
    try:
        run_autogen_approve_demonstration(save_transcript=True)
        dur = time.perf_counter() - t0
        results.append(("Task 14A", "Autogen Review Stage: Grounded Draft Approval", dur, "PASSED"))
        print(f"    [SUCCESS] Task 14 (Case 1) completed in {dur:.2f}s")
    except Exception as e:
        dur = time.perf_counter() - t0
        results.append(("Task 14A", "Autogen Review Stage: Grounded Draft Approval", dur, f"FAILED: {e}"))
        print(f"    [FAILURE] Task 14 (Case 1) failed: {e}")

    # --------------------------------------------------------------------------
    # Step 5: Task 14 Case 2 — Autogen Review Stage: Injected Claim Revision
    # --------------------------------------------------------------------------
    print("\n>>> [5/5] Executing Task 14 (Case 2): Autogen Review Discrepancy Revision...")
    t0 = time.perf_counter()
    try:
        run_autogen_revise_demonstration(save_transcript=True)
        dur = time.perf_counter() - t0
        results.append(("Task 14B", "Autogen Review Stage: Injected Claim Revision", dur, "PASSED"))
        print(f"    [SUCCESS] Task 14 (Case 2) completed in {dur:.2f}s")
    except Exception as e:
        dur = time.perf_counter() - t0
        results.append(("Task 14B", "Autogen Review Stage: Injected Claim Revision", dur, f"FAILED: {e}"))
        print(f"    [FAILURE] Task 14 (Case 2) failed: {e}")

    # --------------------------------------------------------------------------
    # Transcript Audit Assertions
    # --------------------------------------------------------------------------
    print("\n>>> [Audit] Verifying All 5 Part 4 Transcript Artifacts...")
    expected_transcripts = {
        "part4_cache_hit.txt": ["CACHE HIT", "Speedup", "VERIFICATION STATUS: PASSED"],
        "part4_governance_least_autonomy.txt": ["LEAST AUTONOMY", "HIGH RISK", "UnauthorizedToolAccessException"],
        "part4_budget_cap_reject.txt": ["BUDGET CAP EXCEEDED", "HTTP 429", "BudgetExceededException"],
        "part4_autogen_approve.txt": ["RoundRobinGroupChat", "StructuredMessage[ReviewVerdict]", "approved=True"],
        "part4_autogen_revise.txt": ["RoundRobinGroupChat", "StructuredMessage[ReviewVerdict]", "approved=False"],
    }

    audit_passed = True
    for fname, markers in expected_transcripts.items():
        fpath = config.TRANSCRIPTS_DIR / fname
        if not fpath.exists():
            print(f"    [ERROR] Missing transcript: {fname}")
            audit_passed = False
            continue
        text = fpath.read_text(encoding="utf-8")
        if len(text.strip()) == 0:
            print(f"    [ERROR] Empty transcript: {fname}")
            audit_passed = False
            continue
        missing_markers = [m for m in markers if m not in text]
        if missing_markers:
            print(f"    [ERROR] Transcript {fname} missing required markers: {missing_markers}")
            audit_passed = False
        else:
            print(f"    [VERIFIED] {fname} ({len(text)} bytes, markers confirmed)")

    total_dur = time.perf_counter() - start_total
    print("\n" + "=" * 80)
    print("MASTER PART 4 EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Task':<10} | {'Description':<50} | {'Duration':<10} | {'Status'}")
    print("-" * 80)
    all_passed = audit_passed
    for tid, desc, dur, status in results:
        print(f"{tid:<10} | {desc:<50} | {dur:>7.2f}s   | {status}")
        if status != "PASSED":
            all_passed = False

    print("-" * 80)
    print(f"Total Execution Time: {total_dur:.2f}s")
    print(f"Master Verification Status: {'ALL TESTS PASSED' if all_passed else 'TESTS FAILED'}")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_master_part4())

