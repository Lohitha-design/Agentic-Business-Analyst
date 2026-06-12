import logging
from src.llm.base_llm import getllm

logging.basicConfig(level=logging.INFO)

llm = getllm()

def decide(strategy: str, score: float, feedback: str) -> str:
    
    prompt = f"""
    You are a strict decision agent.

    Strategy:
    {strategy}

    Score: {score}
    Feedback: {feedback}

    Rules:
    - If score >= 8 → ACCEPT
    - If improvement possible → REFINE
    - If missing info → RESEARCH

    Return ONLY one word: ACCEPT / REFINE / RESEARCH
    """

    try:
        decision = llm.invoke(prompt).content.strip().upper()
        logging.info(f"Decision agent output: {decision}")

        if decision not in ["ACCEPT", "REFINE", "RESEARCH"]:
            logging.warning("Unexpected decision output, defaulting to REFINE")
            return "REFINE"

        return decision

    except Exception as e:
        logging.error(f"Error in decision agent: {e}")
        return "REFINE"
