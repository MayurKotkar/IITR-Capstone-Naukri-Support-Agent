"""
Task 1: Seeded, Deterministic Job Application Dataset Generator for Naukri.com Domain Support Agent.

This module reuses vocabulary, weights, seed, and salary boundaries from config.py
to generate and validate the JOB_APPLICATIONS dataset deterministically.
"""

import sys
import random
from pathlib import Path

# Fix Windows console UTF-8 output encoding if needed
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path so config can be imported from any working directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config

# One-sentence justification for salary range (required by brief):
SALARY_REASONING = (
    f"The expected salary range of INR {config.MIN_SALARY_INR:,} to INR {config.MAX_SALARY_INR:,} "
    "realistically covers junior roles (Sales Associate, HR Executive) up to senior roles "
    "(Senior Software Engineer, Product Manager) in the Indian recruitment ecosystem."
)


def generate_job_applications(
    seed: int = config.SEED,
    num_records: int = config.NUM_RECORDS,
) -> list[dict]:
    """
    Generate a deterministic list of job application records using configuration parameters.
    
    Each record contains:
    - record_id: str (e.g. 'APP-001')
    - category: str (from config.CATEGORY_WEIGHTS)
    - status: str (from config.STATUS_WEIGHTS)
    - expected_salary_inr: int (between config.MIN_SALARY_INR and config.MAX_SALARY_INR)
    - days_since_created: int (0 to 30)
    - flagged_priority_review: bool (drawn with config.PRIORITY_FLAG_PROBABILITY)
    """
    rng = random.Random(seed)

    categories = list(config.CATEGORY_WEIGHTS.keys())
    cat_weights = list(config.CATEGORY_WEIGHTS.values())

    statuses = list(config.STATUS_WEIGHTS.keys())
    stat_weights = list(config.STATUS_WEIGHTS.values())

    records = []
    for i in range(1, num_records + 1):
        record_id = f"APP-{i:03d}"
        category = rng.choices(categories, weights=cat_weights, k=1)[0]
        status = rng.choices(statuses, weights=stat_weights, k=1)[0]
        
        # Round salary to nearest 10,000 for realistic INR compensation figures
        raw_salary = rng.randint(config.MIN_SALARY_INR // 10000, config.MAX_SALARY_INR // 10000) * 10000
        days_since_created = rng.randint(0, 30)
        flagged_priority_review = rng.random() < config.PRIORITY_FLAG_PROBABILITY

        records.append({
            "record_id": record_id,
            "category": category,
            "status": status,
            "expected_salary_inr": raw_salary,
            "days_since_created": days_since_created,
            "flagged_priority_review": flagged_priority_review,
        })

    return records


def validate_dataset(records: list[dict]) -> dict:
    """
    Validate that the generated dataset satisfies all task criteria:
    - Total records >= 40
    - Every given category has >= 3 records
    - Every given status has >= 1 record
    - Flagged priority review = True percentage is between 10% and 30%
    
    Returns a dictionary of computed summary statistics.
    """
    total_count = len(records)
    if total_count < 40:
        raise AssertionError(f"Validation failed: expected >= 40 records, found {total_count}")

    # Compute category counts
    category_counts = {cat: 0 for cat in config.CATEGORY_WEIGHTS.keys()}
    for r in records:
        category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1

    for cat, count in category_counts.items():
        if count < 3:
            raise AssertionError(f"Validation failed: category '{cat}' has {count} records (minimum required: 3)")

    # Compute status counts
    status_counts = {stat: 0 for stat in config.STATUS_WEIGHTS.keys()}
    for r in records:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    for stat, count in status_counts.items():
        if count < 1:
            raise AssertionError(f"Validation failed: status '{stat}' has {count} records (minimum required: 1)")

    # Compute flagged priority review percentage
    flagged_count = sum(1 for r in records if r["flagged_priority_review"])
    flagged_percentage = (flagged_count / total_count) * 100.0

    if not (10.0 <= flagged_percentage <= 30.0):
        raise AssertionError(
            f"Validation failed: flagged_priority_review percentage {flagged_percentage:.2f}% "
            "is outside the [10%, 30%] range"
        )

    return {
        "total_records": total_count,
        "category_counts": category_counts,
        "status_counts": status_counts,
        "flagged_count": flagged_count,
        "flagged_percentage": flagged_percentage,
    }


# Global dataset instance for importing into other modules/tools
JOB_APPLICATIONS = generate_job_applications()


def format_validation_report(records: list[dict], stats: dict) -> str:
    """Format a clean textual report of the validation and dataset statistics."""
    lines = [
        "=" * 70,
        "PART 1 TASK 1: JOB APPLICATIONS DATASET VALIDATION REPORT",
        "=" * 70,
        f"Track: Naukri.com (Recruitment & HR)",
        f"Deterministic Seed: {config.SEED}",
        f"Total Records: {stats['total_records']} (Requirement: >= 40)",
        f"Salary Range: INR {config.MIN_SALARY_INR:,} to INR {config.MAX_SALARY_INR:,}",
        f"Salary Reasoning: {SALARY_REASONING}",
        "-" * 70,
        "CATEGORY DISTRIBUTION (Requirement: every category >= 3 records):",
    ]
    for cat, count in stats["category_counts"].items():
        lines.append(f"  - {cat:<25}: {count:>2} records [Pass]")

    lines.append("-" * 70)
    lines.append("STATUS DISTRIBUTION (Requirement: every status >= 1 record):")
    for stat, count in stats["status_counts"].items():
        lines.append(f"  - {stat:<25}: {count:>2} records [Pass]")

    lines.append("-" * 70)
    lines.append("PRIORITY REVIEW RATE (Requirement: 10% to 30%):")
    lines.append(
        f"  - Flagged True count   : {stats['flagged_count']} / {stats['total_records']}\n"
        f"  - Flagged percentage   : {stats['flagged_percentage']:.2f}% [Pass: between 10% and 30%]"
    )
    lines.append("-" * 70)
    lines.append("SAMPLE RECORDS (First 5):")
    for r in records[:5]:
        lines.append(
            f"  [{r['record_id']}] {r['category']:<18} | Status: {r['status']:<20} | "
            f"Salary: INR {r['expected_salary_inr']:>9,} | Days: {r['days_since_created']:>2} | "
            f"Priority: {str(r['flagged_priority_review']):<5}"
        )
    lines.append("=" * 70)
    lines.append("OVERALL VALIDATION STATUS: ALL ACCEPTANCE CRITERIA SATISFIED")
    lines.append("=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    stats = validate_dataset(JOB_APPLICATIONS)
    report_text = format_validation_report(JOB_APPLICATIONS, stats)
    print(report_text)

    # Save transcript to naukri-support-agent/transcripts/
    config.TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    transcript_path = config.TRANSCRIPTS_DIR / "part1_dataset_validation.txt"
    transcript_path.write_text(report_text, encoding="utf-8")
    print(f"\n[Saved transcript] {transcript_path}")