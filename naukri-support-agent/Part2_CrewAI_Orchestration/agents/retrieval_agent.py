"""
Retrieval Agent: HR Policy Retrieval Specialist.

Equipped exclusively with the RAG lookup tool to search verified Naukri.com HR policies.
"""

from crewai import Agent
from Part2_CrewAI_Orchestration.tools.rag_tool import rag_lookup_tool


def create_retrieval_agent(llm) -> Agent:
    """Create and return the Retrieval Agent."""
    return Agent(
        role="HR Policy Retrieval Specialist",
        goal="Search and retrieve official Naukri.com HR policy documents using the RAG tool.",
        backstory=(
            "You are an experienced HR policy compliance specialist at Naukri.com. "
            "You have exclusive access to the official policy vector database and "
            "retrieve exact policy clauses to answer employee inquiries accurately."
        ),
        tools=[rag_lookup_tool],
        llm=llm,
        verbose=False,
    )

