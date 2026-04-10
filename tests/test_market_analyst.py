import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.analysts.market_analyst import create_market_analyst


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _CapturingLLM:
    def __init__(self, final_content):
        self.final_content = final_content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return _FakeResponse(self.final_content)


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        indicator = args.get("indicator", "unknown")
        return f"## {indicator} values from 2026-03-01 to 2026-04-01:\n2026-04-01: 1.23"


class MarketAnalystTests(unittest.TestCase):
    def test_market_analyst_backfills_missing_indicators_before_final_report(self):
        llm = _CapturingLLM(
            "# 完整市场报告\n\n已覆盖 SMA、EMA、MACD、RSI、布林带 和 VWMA。"
        )
        node = create_market_analyst(llm)

        partial_result = AIMessage(content="只基于SMA和EMA给出简略报告。")
        state = {
            "company_of_interest": "300308.SZ",
            "trade_date": "2026-04-01",
            "messages": [
                HumanMessage(content="请分析 300308.SZ"),
                AIMessage(content="## close_50_sma values from 2026-03-01 to 2026-04-01:\n2026-04-01: 576.94"),
                AIMessage(content="## close_200_sma values from 2026-03-01 to 2026-04-01:\n2026-04-01: 462.10"),
                AIMessage(content="## close_10_ema values from 2026-03-01 to 2026-04-01:\n2026-04-01: 590.93"),
            ],
        }

        fake_stock_tool = _FakeTool("get_stock_data")
        fake_indicator_tool = _FakeTool("get_indicators")

        with (
            patch(
                "tradingagents.agents.analysts.market_analyst.run_tool_report_chain",
                return_value=(partial_result, "只基于SMA和EMA给出简略报告。"),
            ),
            patch(
                "tradingagents.agents.analysts.market_analyst.get_stock_data",
                fake_stock_tool,
            ),
            patch(
                "tradingagents.agents.analysts.market_analyst.get_indicators",
                fake_indicator_tool,
            ),
        ):
            result = node(state)

        fetched_indicators = {call["indicator"] for call in fake_indicator_tool.calls}
        self.assertIn("macd", fetched_indicators)
        self.assertIn("rsi", fetched_indicators)
        self.assertIn("boll", fetched_indicators)
        self.assertIn("vwma", fetched_indicators)
        self.assertIn("VWMA", result["market_report"])
        self.assertEqual(result["messages"][0].content, result["market_report"])


if __name__ == "__main__":
    unittest.main()
