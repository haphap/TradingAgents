import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.tushare import (
    _build_balance_sheet_summary,
    _build_cashflow_summary,
    _build_income_statement_summary,
    _filter_statement,
    get_balance_sheet,
    get_fundamentals,
)


class _FakePro:
    def stock_basic(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "ts_code": "300750.SZ",
                    "symbol": "300750",
                    "name": "宁德时代",
                    "area": "福建",
                    "industry": "电池",
                    "market": "主板",
                    "list_date": "20180611",
                    "list_status": "L",
                }
            ]
        )

    def daily_basic(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "trade_date": "20260409",
                    "close": 389.99,
                    "turnover_rate": 1.2345,
                    "pe": 24.6524,
                    "pb": 5.2787,
                    "ps": 4.2009,
                    "dv_ratio": 0.85,
                    "total_mv": 177993685.4233,
                    "circ_mv": 146214715.5171,
                }
            ]
        )

    def fina_indicator(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "end_date": "20251231",
                    "roe": 24.7249,
                    "roa": 9.2724,
                    "grossprofit_margin": 26.2728,
                    "netprofit_margin": 18.1227,
                    "debt_to_assets": 61.9393,
                    "ocf_to_or": 31.4432,
                }
            ]
        )

    def balancesheet(self, **kwargs):
        return pd.DataFrame(
            [
                {
                    "end_date": "20251231",
                    "ann_date": "20260315",
                    "update_flag": "1",
                    "total_assets": 974827544000.0,
                    "total_liab": 603801220000.0,
                    "total_hldr_eqy_exc_min_int": 337107747000.0,
                    "total_cur_assets": 638482337000.0,
                    "total_cur_liab": 399625964000.0,
                    "money_cap": 333512927000.0,
                    "accounts_receiv": 76403158000.0,
                    "inventories": 94525980000.0,
                    "contract_liab": 4483962000.0,
                }
            ]
        )


class TushareStatementSummaryTests(unittest.TestCase):
    def test_filter_statement_keeps_latest_updated_row_per_end_date(self):
        df = pd.DataFrame(
            [
                {
                    "end_date": "20250331",
                    "ann_date": "20250415",
                    "update_flag": "0",
                    "total_assets": 100.0,
                },
                {
                    "end_date": "20250331",
                    "ann_date": "20250418",
                    "update_flag": "1",
                    "total_assets": 120.0,
                },
                {
                    "end_date": "20241231",
                    "ann_date": "20250301",
                    "update_flag": "1",
                    "total_assets": 90.0,
                },
            ]
        )

        filtered = _filter_statement(df, "quarterly", "2025-04-20")

        self.assertEqual(filtered["end_date"].tolist(), ["20250331", "20241231"])
        self.assertEqual(filtered.iloc[0]["total_assets"], 120.0)

    def test_balance_sheet_summary_includes_core_totals_and_ratios(self):
        df = pd.DataFrame(
            [
                {
                    "end_date": "20251231",
                    "total_assets": 974827544000.0,
                    "total_liab": 603801220000.0,
                    "total_hldr_eqy_exc_min_int": 337107747000.0,
                    "total_cur_assets": 638482337000.0,
                    "total_cur_liab": 399625964000.0,
                    "money_cap": 333512927000.0,
                }
            ]
        )

        summary = "\n".join(_build_balance_sheet_summary(df))

        self.assertIn("Total Assets: 9748.28亿 CNY", summary)
        self.assertIn("Total Liabilities: 6038.01亿 CNY", summary)
        self.assertIn("Asset-Liability Ratio: 61.94%", summary)
        self.assertIn("Current Ratio: 1.60x", summary)
        self.assertIn("Cash: 3335.13亿 CNY", summary)

    def test_cashflow_and_income_statement_summaries_surface_key_metrics(self):
        cashflow_df = pd.DataFrame(
            [
                {
                    "end_date": "20251231",
                    "n_cashflow_act": 133219982000.0,
                    "n_cashflow_inv_act": -94476299175.3736,
                    "n_cash_flows_fnc_act": -6309607926.27637,
                    "free_cashflow": 87114682973.5947,
                    "c_cash_equ_end_period": 299929459246.0,
                }
            ]
        )
        income_df = pd.DataFrame(
            [
                {
                    "end_date": "20251231",
                    "total_revenue": 423702965000.0,
                    "operate_profit": 89518931000.0,
                    "n_income_attr_p": 72201282000.0,
                    "rd_exp": 22146518000.0,
                },
                {
                    "end_date": "20241231",
                    "total_revenue": 400917983000.0,
                    "operate_profit": 74201022000.0,
                    "n_income_attr_p": 44121205000.0,
                    "rd_exp": 18654002000.0,
                },
            ]
        )

        cashflow_summary = "\n".join(_build_cashflow_summary(cashflow_df))
        income_summary = "\n".join(_build_income_statement_summary(income_df))

        self.assertIn("Operating Cash Flow: 1332.20亿 CNY", cashflow_summary)
        self.assertIn("Free Cash Flow: 871.15亿 CNY", cashflow_summary)
        self.assertIn("Ending Cash and Cash Equivalents: 2999.29亿 CNY", cashflow_summary)
        self.assertIn("Total Revenue: 4237.03亿 CNY", income_summary)
        self.assertIn("Parent Net Income: 722.01亿 CNY", income_summary)
        self.assertIn("Revenue YoY (same period): 5.68%", income_summary)
        self.assertIn("Parent Net Income YoY (same period): 63.64%", income_summary)

    def test_public_tools_include_human_readable_price_and_statement_snapshots(self):
        with patch("tradingagents.dataflows.tushare._get_pro_client", return_value=_FakePro()):
            fundamentals = get_fundamentals("300750.SZ", "2026-04-09")
            balance_sheet = get_balance_sheet("300750.SZ", "quarterly", "2026-04-09")

        self.assertIn("Latest Close Price: 389.99 CNY/share", fundamentals)
        self.assertIn("PE: 24.65x", fundamentals)
        self.assertIn("Total Market Value: 17799.37亿 CNY", fundamentals)
        self.assertIn("Debt to Assets: 61.94%", fundamentals)
        self.assertIn("# Key snapshot", balance_sheet)
        self.assertIn("Total Assets: 9748.28亿 CNY", balance_sheet)
        self.assertIn("Asset-Liability Ratio: 61.94%", balance_sheet)


if __name__ == "__main__":
    unittest.main()
