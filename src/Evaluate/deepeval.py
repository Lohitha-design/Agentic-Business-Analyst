import logging
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

logging.basicConfig(level=logging.INFO)


def evaluate_strategy(strategy: str) -> tuple[float, str]:
    try:

        faithfulness = FaithfulnessMetric()
        relevancy = AnswerRelevancyMetric()

        test_case = LLMTestCase(
            input="Generate a business strategy based on data insights",
            actual_output=strategy
        )

        faithfulness.measure(test_case)
        relevancy.measure(test_case)

        faith_score = faithfulness.score or 0
        rel_score = relevancy.score or 0

        final_score = (faith_score + rel_score) / 2

        feedback = f"""
        Faithfulness: {faithfulness.reason}
        Relevancy: {relevancy.reason}
        """

        logging.info(f"Score: {final_score}")
        logging.info(f"Feedback: {feedback}")

        return final_score, feedback.strip()

    except Exception as e:
        logging.error(f"DeepEval Error: {e}")
        return 5.0, "Evaluation failed. Try refining the strategy."