import os
import phi
import phi.api
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.playground import Playground, serve_playground_app
# Load environment variables
load_dotenv()

## Ensure API ky is set for Phi
phi.api = os.getenv("PHI_API_KEY")

# Ensure API Key is set for Groq (if required)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # Make sure this is set in your .env file

# Validate API key existence
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")

# Web Search Agent
web_search = Agent(
    name="web_search_agent",
    role="Search the web for information",
    model=Groq(id="Deepseek-R1-Distill-Llama-70b"),  # Ensure model ID is correct
    tools=[DuckDuckGo()],
    instructions=["Always include sources",
                  "Do NOT provide responses containing sexual or inappropriate content.",
                  "If a request contains inappropriate content, respond with: 'Warning: This request violates content guidelines.",
                  "Do Not provide responses containing actors and actress and professors and scientists and businessman and businesses and natural resources and places and volcanoes and trees and trenches and oceans and seas out of NVIDIA company or unnecessary content."],
    show_tool_calls=True,
    markdown=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance_agent",
    model=Groq(id="Deepseek-R1-Distill-Llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,company_info=True,
                         historical_prices=True,company_news=True)],
    instructions=["Use table and bar chart to display the data.",
                  "Do NOT provide responses containing sexual or inappropriate content.",
                  "If a request contains inappropriate content, respond with: 'Warning: This request violates content guidelines.",
                  "Do Not provide responses containing actors and actress and professors and scientists and businessman and businesses and natural resources and places and volcanoes and trees and trenches and oceans and seas out of NVIDIA company or unnecessary content."],
    show_tool_calls=False,
    markdown=True
)

# Multi-Agent System
multi_ai_agent = Agent(
    team=[web_search, finance_agent],
    model=Groq(id="Deepseek-R1-Distill-Llama-70b"),
    instructions=[
        "Use the web search agent to find information about the company and the financial agent to find stock details.",
        "Use a table  and bar chat to display stock prices and fundamentals.",
        "Do NOT provide responses containing sexual or inappropriate content.",
        "Do Not provide responses containing actors and actress and professors and scientists and businessman and businesses and natural resources and places and volcanoes and trees and trenches and oceans and seas out of NVIDIA company or unnecessary content.",
        "If a request contains inappropriate content or unnecessary content, respond with: 'Warning: This request violates content guidelines."
    ],
    show_tool_calls=False,
    markdown=True
)

try:
    response = multi_ai_agent.print_response("Please provide the NVIDIA company information", stream=True)
    print(response)  # Print response to console
except Exception as e:
    print(f"Error occurred: {e}")

