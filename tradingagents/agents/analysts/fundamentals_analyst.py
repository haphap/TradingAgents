from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.content_utils import extract_text_content
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_collaboration_stop_instruction,
    get_fundamentals,
    get_income_statement,
    get_language_instruction,
    normalize_chinese_role_terms,
    truncate_for_prompt,
)
from tradingagents.tool_report_utils import run_tool_report_chain


_REQUIRED_FUNDAMENTAL_TOOLS = ("get_fundamentals", "get_balance_sheet", "get_cashflow", "get_income_statement")
_FUNDAMENTAL_TOOL_OUTPUT_MARKERS = {
    "get_fundamentals": ("# tushare fundamentals for ", "# company fundamentals for "),
    "get_balance_sheet": ("# tushare balance sheet for ", "# balance sheet data for "),
    "get_cashflow": ("# tushare cashflow for ", "# cash flow data for "),
    "get_income_statement": ("# tushare income statement for ", "# income statement data for "),
}


def _message_text(message) -> str:
    return extract_text_content(getattr(message, "content", None))


def _collected_fundamentals_context(messages) -> str:
    return "\n\n".join(filter(None, (_message_text(message) for message in messages)))


def _collected_fundamental_tool_names(messages) -> set[str]:
    collected = set()
    for message in messages:
        for tool_call in getattr(message, "tool_calls", None) or []:
            name = tool_call.get("name")
            if name in _REQUIRED_FUNDAMENTAL_TOOLS:
                collected.add(name)

        normalized = _message_text(message).lower()
        for tool_name, markers in _FUNDAMENTAL_TOOL_OUTPUT_MARKERS.items():
            if any(marker in normalized for marker in markers):
                collected.add(tool_name)

    return collected


def _missing_fundamental_tools(messages) -> list[str]:
    collected = _collected_fundamental_tool_names(messages)
    return [tool_name for tool_name in _REQUIRED_FUNDAMENTAL_TOOLS if tool_name not in collected]


def _report_has_full_fundamentals_coverage(report: str) -> bool:
    normalized = (report or "").lower()
    required_keyword_groups = (
        ("balance sheet", "资产负债表"),
        ("income statement", "利润表", "revenue", "营收", "net income", "净利润"),
        ("cash flow", "cashflow", "现金流"),
        ("roe",),
        ("gross margin", "毛利率"),
        ("net margin", "净利率"),
        ("debt to assets", "资产负债率", "leverage", "杠杆"),
        ("free cash flow", "自由现金流"),
        ("growth", "增速", "增长"),
    )
    return all(any(keyword in normalized for keyword in group) for group in required_keyword_groups)


def _fetch_missing_fundamental_data(symbol: str, current_date: str, missing_tools: list[str]) -> str:
    tool_map = {
        get_fundamentals.name: get_fundamentals,
        get_balance_sheet.name: get_balance_sheet,
        get_cashflow.name: get_cashflow,
        get_income_statement.name: get_income_statement,
    }
    tool_args = {
        "get_fundamentals": {"ticker": symbol, "curr_date": current_date},
        "get_balance_sheet": {"ticker": symbol, "freq": "quarterly", "curr_date": current_date},
        "get_cashflow": {"ticker": symbol, "freq": "quarterly", "curr_date": current_date},
        "get_income_statement": {"ticker": symbol, "freq": "quarterly", "curr_date": current_date},
    }

    outputs = []
    for tool_name in missing_tools:
        outputs.append(
            f"### {tool_name}\n{tool_map[tool_name].invoke(tool_args[tool_name])}"
        )
    return "\n\n".join(outputs)


def _rewrite_fundamentals_report(
    llm,
    symbol: str,
    current_date: str,
    instrument_context: str,
    messages,
    missing_tools: list[str],
) -> str:
    existing_context = truncate_for_prompt(
        _collected_fundamentals_context(messages),
        default_limit=24000,
    )
    additional_fundamental_data = _fetch_missing_fundamental_data(symbol, current_date, missing_tools)
    rewrite_prompt = f"""You are a trading assistant writing the final fundamentals report.
Use only the fundamentals and financial statement data supplied below. Do not invent values.

You must explicitly cover all of these sections in the final report:
1. Fundamentals overview and valuation snapshot from `get_fundamentals`.
2. Balance sheet analysis: asset structure, leverage, debt burden, liquidity, and working capital.
3. Income statement analysis: revenue, profit, margin, and earnings quality.
4. Cash flow analysis: operating cash flow, investing cash flow, financing cash flow, free cash flow, and capex implications.
5. Core indicators: ROE, ROA if available, gross margin, net margin, debt-to-assets, OCF-to-revenue if available, plus revenue/profit growth signals from the latest comparable periods when available.

If any statement or metric is unavailable, say that clearly instead of skipping it.
Do not output tool calls. Write a detailed, data-backed markdown report and end with a markdown summary table.
{instrument_context}{get_language_instruction()}

Current date: {current_date}

Existing conversation data:
{existing_context}

Additional statement data fetched to complete the report:
{additional_fundamental_data}
"""
    return normalize_chinese_role_terms(
        extract_text_content(llm.invoke(rewrite_prompt).content)
    )


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Your fundamental analysis must rely only on the structured accounting and financial statement data returned by the fundamentals tools. Do not use news flow, sentiment, rumors, or macro headlines to justify accounting conclusions. Before writing the final report, you must gather data from all of these tools unless a tool explicitly fails: `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, and `get_income_statement`. The final report must explicitly cover the balance sheet, income statement, cash flow statement, and core indicators such as ROE, margins, leverage, cash flow quality, and growth where the data allows it. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction()
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful AI assistant, collaborating with other assistants."
                        " Use the provided tools to progress towards answering the question."
                        " If you are unable to fully answer, that's OK; another assistant with different tools"
                        " will help where you left off. Execute what you can to make progress."
                        + get_collaboration_stop_instruction()
                        + " You have access to the following tools: {tool_names}.\n{system_message}"
                        + "For your reference, the current date is {current_date}. {instrument_context}"
                    ),
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        result, report = run_tool_report_chain(
            prompt_template,
            llm,
            tools,
            state["messages"],
            system_message=system_message,
            tool_names=", ".join([tool.name for tool in tools]),
            current_date=current_date,
            instrument_context=instrument_context,
        )
        report = normalize_chinese_role_terms(report) if report else report

        needs_completion = (
            not getattr(result, "tool_calls", None)
            and (
                not report
                or not _report_has_full_fundamentals_coverage(report)
                or bool(_missing_fundamental_tools([*state["messages"], result]))
            )
        )
        if needs_completion:
            missing_tools = _missing_fundamental_tools([*state["messages"], result])
            completed_report = _rewrite_fundamentals_report(
                llm,
                state["company_of_interest"],
                current_date,
                instrument_context,
                [*state["messages"], result],
                missing_tools,
            )
            if completed_report:
                result = AIMessage(content=completed_report)
                report = completed_report
        elif report and not getattr(result, "tool_calls", None):
            result = AIMessage(content=report)

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
