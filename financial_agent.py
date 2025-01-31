from phi.agent import Agent
from phi.model.huggingface import HuggingFaceChat
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


## web search agent

web_search =Agent(
    name="web_search_agent",
    role="Search the web for information",
    model =HuggingFaceChat(id="deepseek-ai/DeepSeek-V3"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
)

## Finance agent

finance_agent =Agent(
    name="Finance_agent",
    role= ""
    model = HuggingFaceChat(id="deepseek-ai/DeepSeek-V3")
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                         company_news=True)]
)

