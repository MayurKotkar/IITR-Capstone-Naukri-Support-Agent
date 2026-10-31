"""
Response Composer Agent: Senior HR Support Composer.

Pure synthesis agent (NO tools attached). Combines findings from Retrieval and Lookup
agents into a clear, grounded, customer-ready draft answer.
"""

from crewai import Agent


def create_composer_agent(llm) -> Agent:
    """Create and return the Response Composer Agent."""
    return Agent(
        role="Senior HR Support Composer",
        goal="Synthesize clear, professional, and grounded customer responses based on agent findings.",
        backstory=(
            "You are the lead communications specialist at Naukri.com Support. "
            "You take raw policy findings and database records from specialist agents "
            "and compose a cohesive, customer-ready final draft answer."
        ),
        tools=[],  # Pure synthesis agent — no tools wired (least autonomy principle)
        llm=llm,
        verbose=False,
    )

