import logging
from langchain_core.prompts import ChatPromptTemplate 
from src.llm.base_llm import getllm  
from src.tools.search_web import search_web

logging.basicConfig(level=logging.INFO)

llm = getllm()

prompt = ChatPromptTemplate.from_template(
    '''You are an AI research consultant. You are given:
- Refined strategies from another model: {strategy}
- External search results and industry benchmarks: {search_result}

Integrate the research findings into the strategies and produce a final enriched report in the 13-section format:

1. Summary
2. Insights
3. Strategies suggested (enriched with research evidence)
4. Their justification (cite research findings)
5. Strengths
6. Areas of concern
7. Score (from DeepEval)
8. Estimated budget allocation (based on industry benchmarks)
9. Estimated growth if strategies are implemented (grounded in research data)
10. Potential risks (include industry-specific risks)
11. Technological requirements (tools, manpower, hardware/software)
12. Immediate actions
13. How to validate growth (include industry-standard validation methods)

Make sure the strategies are evidence-based and supported by the research results provided.
Return the enriched strategy in a structured format that can be easily parsed for each section. Do not include any extraneous information or explanations outside of the specified sections.'''
)

def research_strategy(strategy: str) -> str:
    
    try:
        search_result = search_web({"query": strategy, "answers": None})

        result = (prompt | llm).invoke({
            "strategy": strategy,
            "search_result": search_result
        })

        logging.info("Research completed successfully")
        return result.content

    except Exception as e:
        logging.error(f"Error during research: {e}")
        raise
