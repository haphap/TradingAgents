import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst


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
    def __init__(self, name, response):
        self.name = name
        self.response = response
        self.calls = []

    def invoke(self, args):
        self.calls.append(args)
        return self.response


class FundamentalsAnalystTests(unittest.TestCase):
    def test_fundamentals_analyst_backfills_missing_statements_before_final_report(self):
        llm = _CapturingLLM(
            "# 完整基本面报告\n\n已覆盖资产负债表、利润表、现金流、ROE、毛利率、净利率、资产负债率、自由现金流和增长。"
        )
        node = create_fundamentals_analyst(llm)

        partial_result = AIMessage(content="只给出笼统基本面结论。")
        state = {
            "company_of_interest": "300750.SZ",
            "trade_date": "2026-04-07",
            "messages": [
                HumanMessage(content="请分析 300750.SZ 的财报和基本面"),
                AIMessage(content="# Tushare fundamentals for 300750.SZ\nROE: 20.1"),
            ],
        }

        fake_fundamentals_tool = _FakeTool(
            "get_fundamentals",
            "# Tushare fundamentals for 300750.SZ\nROE: 20.1\nGross Margin: 24.8",
        )
        fake_balance_sheet_tool = _FakeTool(
            "get_balance_sheet",
            "# Tushare balance sheet for 300750.SZ (quarterly)\nend_date,total_assets,total_liab\n20260331,100,55",
        )
        fake_cashflow_tool = _FakeTool(
            "get_cashflow",
            "# Tushare cashflow for 300750.SZ (quarterly)\nend_date,n_cashflow_act,n_cashflow_inv_act\n20260331,12,-8",
        )
        fake_income_statement_tool = _FakeTool(
            "get_income_statement",
            "# Tushare income statement for 300750.SZ (quarterly)\nend_date,total_revenue,n_income_attr_p\n20260331,25,6",
        )

        with (
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.run_tool_report_chain",
                return_value=(partial_result, "只给出笼统基本面结论。"),
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_fundamentals",
                fake_fundamentals_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_balance_sheet",
                fake_balance_sheet_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_cashflow",
                fake_cashflow_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_income_statement",
                fake_income_statement_tool,
            ),
        ):
            result = node(state)

        self.assertEqual(fake_fundamentals_tool.calls, [])
        self.assertEqual(
            fake_balance_sheet_tool.calls,
            [{"ticker": "300750.SZ", "freq": "quarterly", "curr_date": "2026-04-07"}],
        )
        self.assertEqual(
            fake_cashflow_tool.calls,
            [{"ticker": "300750.SZ", "freq": "quarterly", "curr_date": "2026-04-07"}],
        )
        self.assertEqual(
            fake_income_statement_tool.calls,
            [{"ticker": "300750.SZ", "freq": "quarterly", "curr_date": "2026-04-07"}],
        )
        self.assertIn("自由现金流", result["fundamentals_report"])
        self.assertEqual(result["messages"][0].content, result["fundamentals_report"])

    def test_fundamentals_analyst_rewrites_incomplete_report_when_all_tools_are_already_present(self):
        llm = _CapturingLLM(
            "# Full fundamentals report\n\nCovers balance sheet, income statement, cash flow, ROE, gross margin, net margin, debt to assets, free cash flow, and growth."
        )
        node = create_fundamentals_analyst(llm)

        partial_result = AIMessage(content="Basic overview only.")
        state = {
            "company_of_interest": "300750.SZ",
            "trade_date": "2026-04-07",
            "messages": [
                HumanMessage(content="Analyze 300750.SZ fundamentals"),
                AIMessage(content="# Tushare fundamentals for 300750.SZ\nROE: 20.1"),
                AIMessage(content="# Tushare balance sheet for 300750.SZ (quarterly)\nend_date,total_assets,total_liab"),
                AIMessage(content="# Tushare cashflow for 300750.SZ (quarterly)\nend_date,n_cashflow_act,n_cashflow_inv_act"),
                AIMessage(content="# Tushare income statement for 300750.SZ (quarterly)\nend_date,total_revenue,n_income_attr_p"),
            ],
        }

        fake_fundamentals_tool = _FakeTool("get_fundamentals", "unused")
        fake_balance_sheet_tool = _FakeTool("get_balance_sheet", "unused")
        fake_cashflow_tool = _FakeTool("get_cashflow", "unused")
        fake_income_statement_tool = _FakeTool("get_income_statement", "unused")

        with (
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.run_tool_report_chain",
                return_value=(partial_result, "Basic overview only."),
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_fundamentals",
                fake_fundamentals_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_balance_sheet",
                fake_balance_sheet_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_cashflow",
                fake_cashflow_tool,
            ),
            patch(
                "tradingagents.agents.analysts.fundamentals_analyst.get_income_statement",
                fake_income_statement_tool,
            ),
        ):
            result = node(state)

        self.assertEqual(fake_fundamentals_tool.calls, [])
        self.assertEqual(fake_balance_sheet_tool.calls, [])
        self.assertEqual(fake_cashflow_tool.calls, [])
        self.assertEqual(fake_income_statement_tool.calls, [])
        self.assertIn("free cash flow", result["fundamentals_report"].lower())
        self.assertEqual(result["messages"][0].content, result["fundamentals_report"])


if __name__ == "__main__":
    unittest.main()
