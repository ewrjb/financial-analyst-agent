from google.genai import types
from google.adk.tools import ToolContext
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool # Agent를 tool로 사용할 때 필요
from google.adk.models.lite_llm import LiteLlm
from .sub_agents.data_analyst import data_analyst
from .sub_agents.financial_analyst import financial_analyst
from .sub_agents.news_analyst import news_analyst
from .prompt import PROMPT


MODEL = LiteLlm("openai/gpt-4o")

async def save_advice_report(tool_context: ToolContext, summary: str, ticker: str):
    state = tool_context.state
    data_analyst_result = state.get("data_analyst_result")
    financial_analyst_result = state.get("financial_analyst_result")
    news_analyst_result = state.get("news_analyst_result")
    report = f"""
        # Executive Summary and Advice
        {summary}

        ## Data Analyst Report:
        {data_analyst_result}
         
        ## Financial Analyst Report:
        {financial_analyst_result}

        ## News Analyst Report:
        {news_analyst_result}
    """
    state["report"] = report

    filename = f"{ticker}_investment_advice.md"

    artifact = types.Part(
        inline_data=types.Blob(
            mime_type="text/markdown",
            data=report.encode("utf-8"),
        )
    )

    await tool_context.save_artifact(filename, artifact)

    return {
        "success": True,
    }

financial_advisor = Agent(
    name="FinancialAdvisor",
    instruction=PROMPT,
    model=MODEL,
    tools=[
        AgentTool(agent=financial_analyst),
        AgentTool(agent=news_analyst),
        AgentTool(agent=data_analyst),
        save_advice_report,
    ]
    # sub agent로 처리하는 경우 OpenAI SDK의 Hand-off와 동일.
    # 즉, sub agent로 hand-off된 이후에는 해당 sub agent와 대화하게 됨.
    # 다만, 다른 점은 OpenAI SDK에서는 다시 root agent로 돌아올 수 없지만, ADK에서는 root agent로 돌아올 수 있음.
    # sub_agents=[
    #     financial_analyst,
    #     news_analyst,
    #     data_analyst,
    # ],
)

root_agent = financial_advisor