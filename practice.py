from agno.agent import Agent

from agno.models.openai import OpenAIResponses
from agno.models.groq import Groq

from dotenv import load_dotenv

from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

def build_agent():
    return Agent(
        model = Groq(id='llama-3.3-70b-versatile'),
        markdown = True,
        instructions='you are a helpful and expert travel agent.',
        add_datetime_to_context=True
    )
    
agent = build_agent()

agent.print_response("Who won ipl 2010 ?")