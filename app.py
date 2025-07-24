import os
import streamlit as st
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.tools.yfinance import YFinanceTools

# Load API keys
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Setup agents
@st.cache_resource
def setup_agents():
    web_agent = Agent(
        name="Web Agent",
        role="Search the web for information",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[TavilyTools()],
        instructions="Always include the sources",
        show_tool_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance Agent",
        role="Get financial data",
        model=Gemini(id="gemini-2.0-flash"),
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True
        )],
        instructions="Use tables to display data",
        show_tool_calls=True,
        markdown=True,
    )

    agent_team = Agent(
        team=[web_agent, finance_agent],
        model=Gemini(id="gemini-2.0-flash"),
        instructions=["Always include sources", "Use tables to display data"],
        show_tool_calls=True,
        markdown=True,
    )

    return agent_team

agent_team = setup_agents()

# Streamlit UI
st.set_page_config(page_title="Agentic AI Stock Analyzer", layout="wide")
st.title("📈 Agentic AI Stock Analyzer")

with st.expander("ℹ️ How it works", expanded=False):
    st.markdown("""
    This app uses multiple AI agents to:
    - 🔍 Search the web for up-to-date financial news
    - 📊 Analyze stock data from Yahoo Finance
    - 🧠 Combine insights using a coordinator model powered by Google

    Enter a query like:
    > *"Analyze companies like Tesla, Nvidia and suggest which stocks to buy."*
    """)

user_input = st.text_area("🧠 Enter your financial analysis prompt:", height=150)

if st.button("Run Analysis"):
    if user_input.strip() == "":
        st.warning("Please enter a prompt to analyze.")
    else:
        with st.spinner("🧠 Agents are working on your request..."):
            response = agent_team.run(user_input)
        st.success("✅ Analysis Complete!")
        st.markdown("---")
        st.markdown("### 📋 Agent Response:")
        st.markdown(response.content, unsafe_allow_html=True)
