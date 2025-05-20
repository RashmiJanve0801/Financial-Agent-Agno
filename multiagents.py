from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

web_agent = Agent(
    name = "Web Agent",
    role = "Search the web for information",
    model = Groq(id="qwen-qwq-32b"),
    tools = [DuckDuckGoTools()],
    instructions = "Always include the sources",
    show_tool_calls = True,
    markdown = True,
)

finance_agent = Agent(
    name = "Finance Agent",
    role = "Get financial data",
    model = Gemini(id="gemini-1.5-flash"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions = "Use tables to display data",
    show_tool_calls = True,
    markdown = True,
)

agent_team = Agent(
    team = [web_agent, finance_agent],
    model = Groq(id="qwen-qwq-32b"),
    instructions = ["Always include sources", "Use tables to display data"],
    show_tool_calls = True,
    markdown = True,
)

agent_team.print_response("Analyze companies like Tesla, Nvidia and suggest which stocks to buy ?")