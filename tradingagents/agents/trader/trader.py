import functools
from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_localized_final_proposal_instruction,
    get_output_language,
    normalize_chinese_manager_terms,
    truncate_for_prompt,
)


def _trader_detail_instruction() -> str:
    if get_output_language().strip().lower() in {'chinese', '中文', 'zh', 'zh-cn', 'zh-hans'}:
        return (
            '对于执行计划和风险控制，不能只写“等待支撑”“观察成交量”这类泛化表述而不给解释。'
            '请明确什么算关键支撑或阻力，并优先引用市场报告中的具体类型，例如50日均线、布林中轨、前低或密集成交区；'
            '同时说明成交量改善应相对近5日或20日均量达到什么程度（如“成交量需达到近20日均量的1.3倍以上”）；'
            '以及什么样的催化确认才足以支持加仓、持有、减仓或退出（如“需看到季度订单增速环比提升5个百分点以上”）；'
            '还需要说明盈利验证的具体指标（如毛利率、订单增速、ROE的具体阈值）。'
            '这两部分必须写成完整分析段落，并给出清晰阈值与触发条件。'
        )
    return (
        "For the execution plan and risk controls, do not use generic phrases such as 'wait for support' or 'watch volume' without explanation. "
        "Spell out what counts as key support or resistance by referencing the market report (for example the 50-day moving average, Bollinger mid-band, prior swing low, or dense volume area), "
        "what level of volume recovery counts as improvement (e.g., 'volume must reach 1.3x the 20-day average'), "
        "what specific catalyst confirmation would justify adding, holding, reducing, or exiting (e.g., 'need to see quarterly order growth improve by 5pp QoQ'), "
        "and what earnings-verification metrics matter (e.g., specific thresholds for gross margin, order growth, ROE). "
        "Write these sections as full analytical paragraphs with explicit thresholds and trigger conditions."
    )


def create_trader(llm, memory=None):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = build_instrument_context(company_name)
        investment_plan = truncate_for_prompt(state["investment_plan"])
        market_research_report = truncate_for_prompt(state["market_report"])
        sentiment_report = truncate_for_prompt(state["sentiment_report"])
        news_report = truncate_for_prompt(state["news_report"])
        fundamentals_report = truncate_for_prompt(state["fundamentals_report"])
        research_report = state.get("research_report", "")
        stock_report = state.get("stock_report", "")

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, and social media sentiment. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nIndustry research cross-analysis: {research_report}\nIndividual stock research cross-analysis: {stock_report}\n\nLeverage these insights to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trading agent analyzing market data to make investment decisions. "
                    "Provide a clear thesis, an execution plan, and explicit risk controls. "
                    "If you mention timing in Chinese output, translate it as 时机 or 节奏 instead of leaving the English word. "
                    "For ordinary lists, use Arabic numerals such as 1. 2. 3.; if you use Chinese section headings, keep forms like 一、二、三. "
                    f"{_trader_detail_instruction()} "
                    f"{get_localized_final_proposal_instruction()}{get_language_instruction()}"
                ),
            },
            context,
        ]

        rendered_result = normalize_chinese_manager_terms(
            invoke_structured_or_freetext(
                structured_llm,
                llm,
                messages,
                render_trader_proposal,
                "Trader",
            )
        )

        return {
            "messages": [AIMessage(content=rendered_result)],
            "trader_investment_plan": rendered_result,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
