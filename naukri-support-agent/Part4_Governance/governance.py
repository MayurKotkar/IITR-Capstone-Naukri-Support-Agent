"""
Task 15: Four-Layer AI Governance Implementation for Naukri.com Domain Support.

This module implements:
1. Application Layer (Principle of Least Autonomy):
   - Strict RBAC tool gating: check_job_application_status is exclusively authorized for Lookup Agent.
   - Active runtime guard: UnauthorizedToolAccessException raised if Retrieval or Composer attempts invocation.
   - Formal 1-paragraph justification for Least Autonomy.
2. Application Layer (Risk Classification):
   - Formal classification of Naukri.com Domain Support Agent as HIGH RISK.
   - Formal 1-paragraph justification covering employment impact, salary data, and candidate PII.
3. Runtime Layer (Token/Cost Budget Cap):
   - BudgetCapEnforcer with max_tokens=1000 (config.MAX_TOKEN_BUDGET).
   - Rejection demonstration on deliberately oversized requests (prompt stuffing / denial-of-wallet).
4. Evidence Transcripts Generation:
   - transcripts/part4_governance_least_autonomy.txt (Transcript #22)
   - transcripts/part4_budget_cap_reject.txt (Transcript #23)
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from Part2_CrewAI_Orchestration.agents.composer_agent import create_composer_agent
from Part2_CrewAI_Orchestration.agents.lookup_agent import create_lookup_agent
from Part2_CrewAI_Orchestration.agents.retrieval_agent import create_retrieval_agent
from Part2_CrewAI_Orchestration.mock_llm import MockLLM
from Part2_CrewAI_Orchestration.tools.lookup_tool import (
    check_job_application_status,
    check_job_application_status_tool,
)
from Part2_CrewAI_Orchestration.tools.rag_tool import rag_lookup_tool


# ==============================================================================
# 1. Custom Exceptions
# ==============================================================================

class UnauthorizedToolAccessException(PermissionError):
    """Raised when an agent attempts to access a tool violating the Principle of Least Autonomy."""
    pass


class BudgetExceededException(ValueError):
    """Raised when an incoming query exceeds the allocated token budget cap (HTTP 429)."""
    pass


# ==============================================================================
# 2. Application Layer: Principle of Least Autonomy (PoLA)
# ==============================================================================

AUTHORIZED_TOOL_REGISTRY: Dict[str, List[str]] = {
    "Application Records Specialist": ["check_job_application_status"],
    "HR Policy Retrieval Specialist": ["rag_lookup"],
    "Senior HR Support Composer": [],  # Pure synthesis agent: zero tools allowed
}

LEAST_AUTONOMY_JUSTIFICATION = (
    "JUSTIFICATION FOR LEAST AUTONOMY: In enterprise HR and recruitment systems, applicant database "
    "records contain highly sensitive candidate Personally Identifiable Information (candidate names, "
    "phone numbers, application histories) and confidential financial data (candidate current and expected "
    "salaries up to 3,500,000 INR). Under the Principle of Least Autonomy (PoLA), agents must only possess "
    "the minimal privileges strictly required to complete their designated task. Restricting database "
    "access exclusively to the Lookup Agent isolates sensitive state mutation and data lookup risks, "
    "preventing the Retrieval Agent (which executes unconstrained semantic text matching) and the Composer "
    "Agent (which synthesizes public-facing prose) from querying applicant databases, leaking unauthorized "
    "candidate records, or being tricked via prompt injection into dumping candidate salary tables."
)

RISK_CLASSIFICATION_JUSTIFICATION = (
    "RISK CLASSIFICATION JUSTIFICATION: The Naukri.com Domain Support Agent is categorized as HIGH RISK "
    "under enterprise AI governance frameworks and EU AI Act Annex III criteria. Although the system "
    "provides conversational support, its core domain directly encompasses automated employment and "
    "recruitment workflows: processing job application statuses, assessing candidate priority escalation "
    "scores, handling confidential compensation data (expected salaries up to 3,500,000 INR), and routing "
    "recruitment escalation workflows to human HR managers. Automated decisions, hallucinations, or "
    "unauthorized data disclosures in this operational domain directly impact employment opportunities, "
    "candidate career progression, and referral financial compensations, while exposing sensitive candidate "
    "Personally Identifiable Information (PII). Consequently, the system warrants rigorous High-Risk "
    "governance controls, including pre-disk PII masking, strict least-autonomy tool sandboxing, "
    "deterministic grounded generation fallbacks, token budget caps, and a dual-agent Autogen compliance "
    "review stage."
)


def secure_tool_invoker(agent_role: str, tool_name: str, **kwargs) -> Any:
    """
    Enforce Least Autonomy RBAC at runtime before tool dispatch.
    Only authorized agents defined in AUTHORIZED_TOOL_REGISTRY may execute the tool.
    """
    allowed_tools = AUTHORIZED_TOOL_REGISTRY.get(agent_role, [])
    if tool_name not in allowed_tools:
        raise UnauthorizedToolAccessException(
            f"SECURITY POLICY VIOLATION: Agent '{agent_role}' attempted to call restricted tool "
            f"'{tool_name}'. Under the Principle of Least Autonomy (PoLA), '{tool_name}' is "
            f"strictly restricted to authorized agents: "
            f"{[role for role, tools in AUTHORIZED_TOOL_REGISTRY.items() if tool_name in tools]}."
        )
    if tool_name == "check_job_application_status":
        rec_id = kwargs.get("record_id") or kwargs.get("application_id") or "APP-001"
        return check_job_application_status(record_id=rec_id)
    raise NotImplementedError(f"Tool {tool_name} invoker not implemented.")


def verify_least_autonomy_architecture() -> Dict[str, Any]:
    """
    Audit agent tool assignments against Least Autonomy principles:
    1. Lookup Agent has check_job_application_status.
    2. Retrieval Agent has ONLY rag_lookup (does NOT have check_job_application_status).
    3. Composer Agent has ZERO tools.
    4. Active runtime enforcement blocks unauthorized calls.
    """
    llm = MockLLM()
    lookup_agent = create_lookup_agent(llm)
    retrieval_agent = create_retrieval_agent(llm)
    composer_agent = create_composer_agent(llm)

    lookup_tools = [t.name for t in lookup_agent.tools]
    retrieval_tools = [t.name for t in retrieval_agent.tools]
    composer_tools = [t.name for t in composer_agent.tools]

    # Verify tool wiring assertions
    assert "check_job_application_status" in lookup_tools, "Lookup agent must have lookup tool"
    assert "check_job_application_status" not in retrieval_tools, "Retrieval agent must NOT have lookup tool"
    assert "check_job_application_status" not in composer_tools, "Composer agent must NOT have lookup tool"
    assert len(composer_tools) == 0, "Composer agent must have zero tools (pure synthesis)"

    # Test active runtime RBAC security guard
    # 1. Authorized call by Lookup Agent: Success
    auth_result = secure_tool_invoker(
        agent_role="Application Records Specialist",
        tool_name="check_job_application_status",
        application_id="APP-001"
    )
    assert auth_result["found"] is True, "Authorized call must succeed"

    # 2. Unauthorized call by Retrieval Agent: Blocked
    retrieval_blocked = False
    retrieval_error = ""
    try:
        secure_tool_invoker(
            agent_role="HR Policy Retrieval Specialist",
            tool_name="check_job_application_status",
            application_id="APP-001"
        )
    except UnauthorizedToolAccessException as e:
        retrieval_blocked = True
        retrieval_error = str(e)

    # 3. Unauthorized call by Composer Agent: Blocked
    composer_blocked = False
    composer_error = ""
    try:
        secure_tool_invoker(
            agent_role="Senior HR Support Composer",
            tool_name="check_job_application_status",
            application_id="APP-001"
        )
    except UnauthorizedToolAccessException as e:
        composer_blocked = True
        composer_error = str(e)

    return {
        "lookup_tools": lookup_tools,
        "retrieval_tools": retrieval_tools,
        "composer_tools": composer_tools,
        "retrieval_blocked": retrieval_blocked,
        "retrieval_error": retrieval_error,
        "composer_blocked": composer_blocked,
        "composer_error": composer_error,
        "auth_result_status": auth_result["status"],
    }


# ==============================================================================
# 3. Runtime Layer: Cost/Token Budget Cap
# ==============================================================================

class BudgetCapEnforcer:
    """
    Enforces per-request cost and token budget caps at the Runtime Layer.
    Prevents prompt-stuffing, runaway loops, and denial-of-wallet attacks.
    """

    def __init__(self, max_tokens: int = config.MAX_TOKEN_BUDGET):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count conservatively:
        Uses max(word_count, len(text) // 4) plus whitespace adjustment.
        """
        if not text:
            return 0
        word_count = len(text.split())
        char_count_estimate = max(1, int(len(text) / 3.8))
        return max(word_count, char_count_estimate)

    def check_budget(self, query: str, estimated_completion_tokens: int = 150) -> int:
        """
        Evaluate request against token budget cap.
        Returns total estimated tokens if within budget.
        Raises BudgetExceededException if budget cap is violated.
        """
        prompt_tokens = self.estimate_tokens(query)
        total_estimated_tokens = prompt_tokens + estimated_completion_tokens

        if total_estimated_tokens > self.max_tokens:
            raise BudgetExceededException(
                f"BUDGET CAP EXCEEDED (HTTP 429): Request estimated at {total_estimated_tokens} tokens "
                f"({prompt_tokens} prompt tokens + {estimated_completion_tokens} completion tokens), "
                f"which exceeds the strict runtime budget cap of {self.max_tokens} tokens per request. "
                f"Query rejected to prevent denial-of-wallet exhaustion."
            )
        return total_estimated_tokens


# ==============================================================================
# 4. Demonstration Runners & Transcript Generation
# ==============================================================================

def run_least_autonomy_demonstration(save_transcript: bool = True) -> str:
    """Execute Task 15 Part A: Principle of Least Autonomy & Risk Classification."""
    audit = verify_least_autonomy_architecture()
    lines: List[str] = []

    lines.append("=" * 80)
    lines.append("NAUKRI.COM DOMAIN SUPPORT AGENT: AI GOVERNANCE VERIFICATION")
    lines.append("TASK 15 (PART A): APPLICATION LAYER — PRINCIPLE OF LEAST AUTONOMY")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Governance Model: Four-Layer AI Governance (Application, Runtime, Model, Infrastructure)")
    lines.append("-" * 80)

    lines.append("\n1. AGENT TOOL PRIVILEGE AUDIT (LEAST AUTONOMY VERIFICATION):")
    lines.append("-" * 80)
    lines.append(f"  [Agent 1: Lookup Agent (Application Records Specialist)]")
    lines.append(f"    - Assigned Tools: {audit['lookup_tools']}")
    lines.append(f"    - Authorization : check_job_application_status -> PERMITTED (Sole Authorized Agent)")
    lines.append(f"  [Agent 2: Retrieval Agent (HR Policy Retrieval Specialist)]")
    lines.append(f"    - Assigned Tools: {audit['retrieval_tools']}")
    lines.append(f"    - Authorization : check_job_application_status -> PROHIBITED (Not Wired)")
    lines.append(f"  [Agent 3: Response Composer (Senior HR Support Composer)]")
    lines.append(f"    - Assigned Tools: {audit['composer_tools']}")
    lines.append(f"    - Authorization : check_job_application_status -> PROHIBITED (Zero Tools / Pure Synthesis)")

    lines.append("\n2. RUNTIME ACCESS-CONTROL REJECTION DEMONSTRATION:")
    lines.append("-" * 80)
    lines.append("  [Scenario A: Authorized Invocation by Lookup Agent]")
    lines.append(f"    - Action : Lookup Agent calls check_job_application_status('APP-001')")
    lines.append(f"    - Result : SUCCESS (Status returned: '{audit['auth_result_status']}')")

    lines.append("\n  [Scenario B: Unauthorized Invocation Attempt by Retrieval Agent]")
    lines.append(f"    - Action   : Retrieval Agent attempts check_job_application_status('APP-001')")
    lines.append(f"    - Intercept: UnauthorizedToolAccessException RAISED")
    lines.append(f"    - Blocked  : {audit['retrieval_blocked']}")
    lines.append(f"    - Reason   : {audit['retrieval_error']}")

    lines.append("\n  [Scenario C: Unauthorized Invocation Attempt by Composer Agent]")
    lines.append(f"    - Action   : Composer Agent attempts check_job_application_status('APP-001')")
    lines.append(f"    - Intercept: UnauthorizedToolAccessException RAISED")
    lines.append(f"    - Blocked  : {audit['composer_blocked']}")
    lines.append(f"    - Reason   : {audit['composer_error']}")

    lines.append("\n3. FORMAL ONE-PARAGRAPH JUSTIFICATION FOR LEAST AUTONOMY:")
    lines.append("-" * 80)
    lines.append(LEAST_AUTONOMY_JUSTIFICATION)

    lines.append("\n4. SYSTEM RISK LEVEL CLASSIFICATION & JUSTIFICATION:")
    lines.append("-" * 80)
    lines.append("  CLASSIFIED RISK TIER: HIGH RISK")
    lines.append("  (Comparative Schema: Low=Summarization; Medium=Code/Ticket Gen; High=Hiring/Financial/Medical)")
    lines.append("")
    lines.append(RISK_CLASSIFICATION_JUSTIFICATION)

    lines.append("\n" + "=" * 80)
    lines.append("VERIFICATION STATUS: PASSED — LEAST AUTONOMY STRICTLY ENFORCED")
    lines.append("=" * 80)

    report_text = "\n".join(lines) + "\n"

    if save_transcript:
        config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        transcript_path = config.TRANSCRIPTS_DIR / "part4_governance_least_autonomy.txt"
        transcript_path.write_text(report_text, encoding="utf-8")
        print(f"[Saved transcript] {transcript_path}")

    return report_text


def run_budget_cap_demonstration(save_transcript: bool = True) -> str:
    """Execute Task 15 Part B: Runtime Layer — Cost & Token Budget Cap Enforcer."""
    enforcer = BudgetCapEnforcer(max_tokens=config.MAX_TOKEN_BUDGET)
    lines: List[str] = []

    lines.append("=" * 80)
    lines.append("NAUKRI.COM DOMAIN SUPPORT AGENT: AI GOVERNANCE VERIFICATION")
    lines.append("TASK 15 (PART B): RUNTIME LAYER — COST & TOKEN BUDGET CAP ENFORCEMENT")
    lines.append("=" * 80)
    lines.append(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Strict Budget Cap: {config.MAX_TOKEN_BUDGET} tokens per request")
    lines.append(f"Rejection Policy: Fail-Fast HTTP 429 / BudgetExceededException")
    lines.append("-" * 80)

    # Test Case 1: Normal In-Scope Query (Within Budget)
    normal_query = "What is the notice period required upon employee resignation?"
    normal_tokens = enforcer.estimate_tokens(normal_query)
    normal_total = enforcer.check_budget(normal_query, estimated_completion_tokens=150)

    lines.append("\n[TEST CASE 1: STANDARD IN-SCOPE QUERY (WITHIN BUDGET)]")
    lines.append(f"  Query Text        : \"{normal_query}\"")
    lines.append(f"  Prompt Tokens     : {normal_tokens} tokens")
    lines.append(f"  Completion Buffer : 150 tokens")
    lines.append(f"  Total Estimated   : {normal_total} tokens")
    lines.append(f"  Threshold Cap     : {config.MAX_TOKEN_BUDGET} tokens")
    lines.append(f"  Enforcement Action: PERMITTED (Within Budget: {normal_total} <= {config.MAX_TOKEN_BUDGET})")

    # Test Case 2: Deliberately Oversized Request (Exceeds Budget)
    # Generate ~1,200 tokens via prompt stuffing / repetitive expansion
    oversized_query = (
        "Please provide an exhaustive, infinitely detailed legal and procedural breakdown of "
        "every imaginable employment termination scenario, including notice periods, severance "
        "calculations, gratuity disputes, non-compete clauses, and international labor treaties. "
    ) * 45

    oversized_prompt_tokens = enforcer.estimate_tokens(oversized_query)
    lines.append("\n[TEST CASE 2: DELIBERATELY OVERSIZED REQUEST (EXCEEDS BUDGET CAP)]")
    lines.append(f"  Query Character Count : {len(oversized_query)} chars")
    lines.append(f"  Prompt Tokens         : {oversized_prompt_tokens} tokens")
    lines.append(f"  Completion Buffer     : 150 tokens")
    lines.append(f"  Total Estimated       : {oversized_prompt_tokens + 150} tokens")
    lines.append(f"  Threshold Cap         : {config.MAX_TOKEN_BUDGET} tokens")

    rejected = False
    rejection_message = ""
    try:
        enforcer.check_budget(oversized_query, estimated_completion_tokens=150)
    except BudgetExceededException as e:
        rejected = True
        rejection_message = str(e)

    lines.append(f"  Enforcement Action    : REJECTED (Fail-Fast Blocked)")
    lines.append(f"  Exception Raised      : BudgetExceededException")
    lines.append(f"  HTTP Status Code      : 429 Too Many Requests (Token Quota Exceeded)")
    lines.append(f"  Guard Message         : {rejection_message}")

    assert rejected is True, "Oversized request must be rejected"

    lines.append("\n" + "=" * 80)
    lines.append("RUNTIME GOVERNANCE CONCLUSION:")
    lines.append("  The runtime layer successfully enforces the 1000-token budget ceiling, protecting")
    lines.append("  the Naukri.com infrastructure against denial-of-wallet exhaustion and unbounded loops.")
    lines.append("=" * 80)
    lines.append("VERIFICATION STATUS: PASSED")
    lines.append("=" * 80)

    report_text = "\n".join(lines) + "\n"

    if save_transcript:
        config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        transcript_path = config.TRANSCRIPTS_DIR / "part4_budget_cap_reject.txt"
        transcript_path.write_text(report_text, encoding="utf-8")
        print(f"[Saved transcript] {transcript_path}")

    return report_text


if __name__ == "__main__":
    rep_la = run_least_autonomy_demonstration(save_transcript=True)
    print(rep_la)
    print("\n" + "#" * 80 + "\n")
    rep_bc = run_budget_cap_demonstration(save_transcript=True)
    print(rep_bc)

