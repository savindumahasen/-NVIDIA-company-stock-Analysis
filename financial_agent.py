import os
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

# Finance Agent
finance_agent = Agent(
    name="Finance_agent",
    model=Groq(id="Deepseek-R1-Distill-Llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                         company_news=True)],
    instructions=["Use table to display the data."],
    show_tool_calls=True,
    markdown=True
)

# Multi-Agent System
multi_ai_agent = Agent(
    team=[web_search, finance_agent],
    model=Groq(id="Deepseek-R1-Distill-Llama-70b"),
    instructions=[
        "Use the web search agent to find information about the company and the financial agent to find stock details.",
        "Use a table to display stock prices and fundamentals."
    ],
    show_tool_calls=True,
    markdown=True
)

# Request NVIDIA stock price
try:
    response = multi_ai_agent.print_response("Please share the news for NVIDIA company", stream=True)
    print(response)  # Print response to console
except Exception as e:
    print(f"Error occurred: {e}")
