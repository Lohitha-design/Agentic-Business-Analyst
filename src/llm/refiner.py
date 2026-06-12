import logging
from langchain_core.prompts import ChatPromptTemplate
from src.llm.base_llm import getllm

logging.basicConfig(level=logging.INFO)

llm = getllm()

prompt = ChatPromptTemplate.from_template(
    '''You are an AI strategy refiner. Your task is to take:
- Strategies suggested by another model: {strategy}
- Feedback and scoring results from DeepEval: {feedback}

Refine the strategies to make them more actionable, realistic, and aligned with the feedback.


📑 Sections to include:
1. Summary (short paragraph only)  
2. Insights (bullet points)  
3. Refined strategies suggested (bullet points)  
4. Their justification (bullet points)  
5. Strengths (bullet points)  
6. Areas of concern (bullet points)  
7. Score (bullet points, use DeepEval feedback)  
8. Estimated budget allocation (bullet points)  
9. Estimated growth if refined strategies are implemented (bullet points)  
10. Potential risks (bullet points)  
11. Technological requirements (bullet points)  
12. Immediate actions (bullet points)  
13. Validation methods (bullet points)  

Ensure refinements address weaknesses highlighted in the feedback while preserving strong points.
Return the final output strictly in this 13‑section format.
Do not include any extraneous information or explanations outside of the specified sections.'''
)

def refine_strategy(strategy: str, feedback: str) -> str:
   
    try:
        result = (prompt | llm).invoke({
            "strategy": strategy,
            "feedback": feedback
        })
        logging.info("Strategy refinement completed successfully")
        return result.content
    except Exception as e:
        logging.error(f"Error refining strategy: {e}")
        raise
