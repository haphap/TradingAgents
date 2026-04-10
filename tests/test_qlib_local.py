import unittest
from unittest.mock import patch

import pandas as pd

from tradingagents.dataflows.exceptions import DataVendorUnavailable
from tradingagents.dataflows import qlib_local


class QlibLocalTests(unittest.TestCase):
    def test_load_ohlcv_keeps_qlib_prices_and_drops_incomplete_rows(self):
        idx = pd.to_datetime(["2026-03-07", "2026-03-10", "2026-03-11"])
        feature_map = {
            "open": pd.Series([1.9, 2.1, 2.3], index=idx, name="open"),
            "high": pd.Series([2.1, 2.3], index=idx[:2], name="high"),
            "low": pd.Series([1.8, 2.0, 2.2], index=idx, name="low"),
            "close": pd.Series([2.0, 2.2], index=idx[:2], name="close"),
            "volume": pd.Series([1000.0, 1200.0], index=idx[:2], name="volume"),
            "amount": pd.Series([500.0, 600.0], index=idx[:2], name="amount"),
        }

        def fake_read_feature(_instrument, field, _start, _end):
            return feature_map.get(field, pd.Series(dtype="float64", name=field))

        with patch("tradingagents.dataflows.qlib_local._read_feature", side_effect=fake_read_feature):
            df = qlib_local._load_ohlcv("sz300750", "2026-03-07", "2026-03-11")

        self.assertEqual(list(df.index.strftime("%Y-%m-%d")), ["2026-03-07", "2026-03-10"])
        self.assertAlmostEqual(df.loc[pd.Timestamp("2026-03-07"), "close"], 2.0)
        self.assertAlmostEqual(df.loc[pd.Timestamp("2026-03-10"), "open"], 2.1)
        self.assertAlmostEqual(df.loc[pd.Timestamp("2026-03-10"), "volume"], 1200.0)
        self.assertAlmostEqual(df.loc[pd.Timestamp("2026-03-07"), "amount"], 500.0)
        self.assertNotIn("factor", df.columns)
        self.assertNotIn("adjclose", df.columns)

    def test_get_stock_raises_when_local_data_is_stale(self):
        idx = pd.to_datetime(["2026-02-24"])
        df = pd.DataFrame(
            {
                "open": [360.0],
                "high": [365.0],
                "low": [355.0],
                "close": [362.0],
                "volume": [1_000_000.0],
                "amount": [362_000_000.0],
            },
            index=idx,
        )
        df.index.name = "Date"

        with (
            patch("tradingagents.dataflows.qlib_local._load_ohlcv", return_value=df),
            patch(
                "tradingagents.dataflows.qlib_local._expected_last_trading_day",
                return_value=pd.Timestamp("2026-03-10"),
            ),
        ):
            with self.assertRaises(DataVendorUnavailable):
                qlib_local.get_stock("300750.SZ", "2026-01-01", "2026-03-10")

    def test_load_price_frame_raises_when_local_data_is_stale(self):
        idx = pd.to_datetime(["2026-02-24"])
        df = pd.DataFrame(
            {
                "open": [360.0],
                "high": [365.0],
                "low": [355.0],
                "close": [362.0],
                "volume": [1_000_000.0],
            },
            index=idx,
        )
        df.index.name = "Date"

        with (
            patch("tradingagents.dataflows.qlib_local._load_ohlcv", return_value=df),
            patch(
                "tradingagents.dataflows.qlib_local._expected_last_trading_day",
                return_value=pd.Timestamp("2026-03-10"),
            ),
        ):
            with self.assertRaises(DataVendorUnavailable):
                qlib_local._load_price_frame("300750.SZ", "2026-03-10")


if __name__ == "__main__":
    unittest.main()
