import logging
from langchain_core.prompts import ChatPromptTemplate 
from src.llm.base_llm import getllm

logging.basicConfig(level=logging.INFO)

llm = getllm()

prompt = ChatPromptTemplate.from_template(
    '''You are an AI business analyst. Using the structured data provided (unsupervised analysis results : {unsupervised_data}, regression results : {insights}, correlation results : {features}),
      generate a comprehensive business report in the following format:

1. Summary
2. Insights
3. Strategies suggested
4. Their justification
5. Strengths of the given data analysis (the already good things in the business data, not data quality)
6. Areas of concern (the things that need attention in the business data, not data quality)
7. Score (leave blank, this will be filled separately)
8. Estimated budget allocation
9. Estimated growth if the suggestions were implemented
10. Potential risks
11. Technological requirements (manpower, hardware, software)
12. Immediate actions to be taken
13. How to validate the growth

Use the structured inputs directly to justify your points. Keep the output concise, clear, and business-oriented.
Return the report in a structured format that can be easily parsed for each section. Do not include any extraneous information or explanations outside of the specified sections.'''
)

def generate_strategy(insights: dict, features: dict, unsupervised_data: dict) -> str:
    
    try:
        result = (prompt | llm).invoke({
            "insights": insights,
            "features": features,
            "unsupervised_data": unsupervised_data
        })
        logging.info("Business strategy generated successfully")
        return result.content
    except Exception as e:
        logging.error(f"Error generating strategy: {e}")
        raise
