from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.content_utils import extract_text_content
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_collaboration_stop_instruction,
    get_indicators,
    get_language_instruction,
    get_stock_data,
    truncate_for_prompt,
)
from tradingagents.tool_report_utils import run_tool_report_chain


_REQUIRED_MARKET_INDICATORS = {
    "close_50_sma": 90,
    "close_200_sma": 260,
    "close_10_ema": 45,
    "macd": 60,
    "macds": 60,
    "macdh": 60,
    "rsi": 60,
    "boll": 60,
    "boll_ub": 60,
    "boll_lb": 60,
    "atr": 60,
    "vwma": 60,
}


def _message_text(message) -> str:
    return extract_text_content(getattr(message, "content", None))


def _collected_market_context(messages) -> str:
    return "\n\n".join(filter(None, (_message_text(message) for message in messages)))


def _missing_market_indicators(messages) -> list[str]:
    conversation = _collected_market_context(messages).lower()
    missing = []
    for indicator in _REQUIRED_MARKET_INDICATORS:
        if f"## {indicator} values" not in conversation:
            missing.append(indicator)
    return missing


def _report_has_full_market_coverage(report: str) -> bool:
    normalized = (report or "").lower()
    required_keyword_groups = (
        ("sma", "ema"),
        ("macd",),
        ("rsi",),
        ("boll", "布林"),
        ("vwma", "成交量加权移动平均线", "volume weighted"),
    )
    return all(any(keyword in normalized for keyword in group) for group in required_keyword_groups)


def _fetch_missing_indicator_data(symbol: str, current_date: str, missing_indicators: list[str]) -> str:
    outputs = []
    for indicator in missing_indicators:
        outputs.append(
            get_indicators.invoke(
                {
                    "symbol": symbol,
                    "indicator": indicator,
                    "curr_date": current_date,
                    "look_back_days": _REQUIRED_MARKET_INDICATORS[indicator],
                }
            )
        )
    return "\n\n".join(outputs)


def _rewrite_market_report(llm, symbol: str, current_date: str, instrument_context: str, messages, missing_indicators: list[str]) -> str:
    existing_context = truncate_for_prompt(_collected_market_context(messages), default_limit=24000)
    additional_indicator_data = _fetch_missing_indicator_data(symbol, current_date, missing_indicators)
    rewrite_prompt = f"""You are a trading assistant writing the final market technical report.
Use only the stock data and indicator outputs supplied below. Do not invent values.

You must explicitly cover all of these categories in the final report:
1. Trend and moving averages: close_50_sma, close_200_sma, close_10_ema
2. Momentum: macd, macds, macdh, rsi
3. Volatility: boll, boll_ub, boll_lb, atr
4. Volume confirmation: vwma

If any indicator is unavailable, state that clearly instead of skipping the section.
Do not output tool calls. Write a detailed, data-backed markdown report and end with a markdown summary table.
{instrument_context}{get_language_instruction()}

Current date: {current_date}

Existing conversation data:
{existing_context}

Additional indicator data fetched to complete the report:
{additional_indicator_data}
"""
    return extract_text_content(llm.invoke(rewrite_prompt).content)


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your price and technical analysis must rely only on the data returned by `get_stock_data` and `get_indicators`. Do not infer price action, volume behavior, momentum, or technical signals from company news, macro news, or social sentiment. You must produce a comprehensive technical report, not a minimal one. Unless a tool explicitly fails, cover all of the following indicators in the final report:

- close_50_sma
- close_200_sma
- close_10_ema
- macd
- macds
- macdh
- rsi
- boll
- boll_ub
- boll_lb
- atr
- vwma

Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
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

        needs_completion = (
            not getattr(result, "tool_calls", None)
            and (
                not report
                or not _report_has_full_market_coverage(report)
                or bool(_missing_market_indicators([*state["messages"], result]))
            )
        )
        if needs_completion:
            missing_indicators = _missing_market_indicators([*state["messages"], result])
            completed_report = _rewrite_market_report(
                llm,
                state["company_of_interest"],
                current_date,
                instrument_context,
                [*state["messages"], result],
                missing_indicators,
            )
            if completed_report:
                result = AIMessage(content=completed_report)
                report = completed_report

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
