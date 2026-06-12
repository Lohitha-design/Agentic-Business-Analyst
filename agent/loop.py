from src.llm.stratergy import generate_strategy
from src.llm.refiner import refine_strategy
from src.llm.decision import decide
from src.Evaluate.deepeval import evaluate_strategy
from src.tools.research_tool import research_strategy


def run_agent(insights: dict, features: dict, unsupervised_data: dict) -> str:

    strategy = generate_strategy(insights, features, unsupervised_data)

    for iteration in range(5):
        score, feedback = evaluate_strategy(strategy)

        if score >= 8:
            return strategy

        decision = decide(strategy, score, feedback)

        if decision == "ACCEPT":
            return strategy

        elif decision == "REFINE":
            strategy = refine_strategy(strategy, feedback)

        elif decision == "RESEARCH":
            extra_info = research_strategy(strategy)  
            strategy = refine_strategy(strategy, extra_info)

    
    return strategy