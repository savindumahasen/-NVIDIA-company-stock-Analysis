from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv

load_dotenv()

## Web Search Agent
web_search = Agent(
    name="web_search_agent",
    role="Search the web for information",
    model=Groq(id="Llama3-8b-8192"),  # FIXED
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

## Finance Agent
finance_agent = Agent(
    name="Finance_agent",
    model=Groq(id="Llama3-8b-8192"),  # FIXED
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                         company_news=True)],
    instructions=["Use table to display the data"],
    show_tool_calls=False,
    markdown=True
)

## Multi-Agent System
multi_ai_agent = Agent(
    team=[web_search, finance_agent],
    model=Groq(id="Llama3-8b-8192"), # Replace OpenAI with Hugging Face
    instructions=[
        "Use the web search agent to find information about the company and the financial agent to find information about the stock.",
        "Use a table to display the data."
    ],
    show_tool_calls=False,
    markdown=True
)


multi_ai_agent.print_response("Please provide  stock item of NVIDIA",stream=True)
