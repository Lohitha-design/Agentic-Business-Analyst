from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

def getllm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=0.7
    )

