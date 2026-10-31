"""
Lookup Agent: Application Records Specialist.

Equipped exclusively with the check_job_application_status tool to query applicant records
and compute escalation scores (enforcing the principle of least autonomy).
"""

from crewai import Agent
from Part2_CrewAI_Orchestration.tools.lookup_tool import check_job_application_status_tool


def create_lookup_agent(llm) -> Agent:
    """Create and return the Lookup Agent."""
    return Agent(
        role="Application Records Specialist",
        goal="Look up job application records and calculate escalation metrics from the database.",
        backstory=(
            "You are a senior recruiter operations specialist at Naukri.com who tracks "
            "candidate application statuses, expected salaries, and priority escalation scores."
        ),
        tools=[check_job_application_status_tool],
        llm=llm,
        verbose=False,
    )

