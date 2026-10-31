import os
import re
from pathlib import Path

# ============================================================
# 1. Environment variables — set BEFORE importing other libraries
# ============================================================

os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ============================================================
# 2. Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_DB_PATH = str(BASE_DIR / "naukri-support-agent/Part1_Dataset_RAG/chroma_db")
KNOWLEDGE_BASE_DIR = BASE_DIR / "naukri-support-agent/Part1_Dataset_RAG/knowledge_base"
TRANSCRIPTS_DIR = BASE_DIR / "naukri-support-agent/transcripts"
LOGS_DIR = BASE_DIR / "naukri-support-agent/logs"
AUDIT_LOG_PATH = LOGS_DIR / "audit.jsonl"

# ============================================================
# 3. Execution mode
# ============================================================

USE_MOCK_LLM = True  # No paid LLM API required for graded execution

# ============================================================
# 4. Dataset defaults
# ============================================================

SEED = 42
NUM_RECORDS = 50

MIN_SALARY_INR = 300_000
MAX_SALARY_INR = 3_500_000

# Weighted sampling defaults (percentages)
CATEGORY_WEIGHTS = {
    "Software Engineer": 30,
    "Data Analyst": 25,
    "Product Manager": 20,
    "HR Executive": 15,
    "Sales Associate": 10,
}

STATUS_WEIGHTS = {
    "Applied": 40,
    "Screening": 25,
    "Interview Scheduled": 15,
    "Offered": 10,
    "Rejected": 10,
}

PRIORITY_FLAG_PROBABILITY = 0.20

# Escalation scoring parameters (Part 2 Task 6)
# escalation_score = w_priority * flag + w_recency * (1 - days / 30)
ESCALATION_WEIGHT_PRIORITY = 0.60
ESCALATION_WEIGHT_RECENCY = 0.40
# Threshold justified by dataset score distribution:
# Non-flagged records max out at 0.40; flagged records with recency >= 0.125 score >= 0.65.
ESCALATION_THRESHOLD = 0.65

# ============================================================
# 5. RAG and vector database
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

FIXED_CHUNK_COLLECTION = "fixed_size_chunks"
SENTENCE_CHUNK_COLLECTION = "sentence_chunks"

RAG_TOP_K = 3

# Empirically calibrated similarity threshold (Part 1 Task 4)
# In-scope range: [0.5145, 0.7599], Out-of-scope range: [0.1159, 0.2525]
RAG_SIMILARITY_THRESHOLD = 0.38

# Recommended collection based on Task 5 empirical precision/recall evaluation:
# Sentence-based chunking achieved 0.7667 precision vs 0.7000 for fixed-size (both recall = 1.0000).
RECOMMENDED_RAG_COLLECTION = SENTENCE_CHUNK_COLLECTION

# ============================================================
# 6. Guardrail patterns — Indian and international phone numbers
# ============================================================

# Indian mobile numbers: optional +91/0 prefix, then 10 digits
INDIAN_PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:(?:\+91|0091|91)[ -]?)?(?:0[ -]?)?[6-9]\d{9}(?!\d)"
)

# International numbers: a + country code followed by 7–14
# additional digits, allowing common separators.
INTERNATIONAL_PHONE_PATTERN = re.compile(
    r"(?<!\w)\+[1-9]\d{0,2}(?:[ .()-]?\d){7,14}(?!\w)"
)

# ============================================================
# 7. Governance
# ============================================================

MAX_TOKEN_BUDGET = 1000

# ============================================================
# 8. Create standard project directories if missing
# ============================================================

KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
Path(CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)