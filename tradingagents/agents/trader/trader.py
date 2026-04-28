import functools
from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.structured import invoke_structured_or_freetext
from tradingagents.content_utils import extract_text_content
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_localized_final_proposal_instruction,
    truncate_for_prompt,
)


def create_trader(llm, memory=None):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = truncate_for_prompt(state["investment_plan"])
        market_research_report = truncate_for_prompt(state["market_report"])
        sentiment_report = truncate_for_prompt(state["sentiment_report"])
        news_report = truncate_for_prompt(state["news_report"])
        fundamentals_report = truncate_for_prompt(state["fundamentals_report"])

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Provide a clear thesis, an execution plan, and explicit risk controls. "
                    "If you mention timing in Chinese output, translate it as 时机 or 节奏 instead of leaving the English word. "
                    "Use Arabic numerals such as 1. 2. 3. for any numbered items. "
                    f"{get_localized_final_proposal_instruction()}{get_language_instruction()}"
                ),
            },
            context,
        ]

        result = invoke_structured_or_freetext(llm, messages, TraderProposal)
        if isinstance(result, TraderProposal):
            rendered_result = render_trader_proposal(result)
        else:
            rendered_result = extract_text_content(result)

        return {
            "messages": [AIMessage(content=rendered_result)],
            "trader_investment_plan": rendered_result,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
