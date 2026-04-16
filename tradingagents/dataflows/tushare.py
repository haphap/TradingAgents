from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Callable

import pandas as pd
from stockstats import wrap

from .exceptions import DataVendorUnavailable


_SUPPORTED_EXCHANGES = {"SH", "SZ", "BJ", "HK"}
_SUFFIX_MAP = {
    "SH": "SH",
    "SS": "SH",
    "SSE": "SH",
    "SZ": "SZ",
    "SZSE": "SZ",
    "BJ": "BJ",
    "BSE": "BJ",
    "HK": "HK",
    "HKG": "HK",
    "SEHK": "HK",
}

_A_SHARE_EXCHANGES = {"SH", "SZ", "BJ"}


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _to_api_date(date_str: str) -> str:
    return _parse_date(date_str).strftime("%Y%m%d")


def _classify_market(ts_code: str) -> str:
    if "." in ts_code:
        suffix = ts_code.rsplit(".", 1)[1]
        if suffix in _A_SHARE_EXCHANGES:
            return "a_share"
        if suffix == "HK":
            return "hk"
    return "us"


def _normalize_ts_code(symbol: str) -> str:
    raw = symbol.strip().upper()

    if "." in raw:
        code, suffix = raw.split(".", 1)
        suffix = _SUFFIX_MAP.get(suffix, suffix)
        if suffix in _A_SHARE_EXCHANGES and code.isdigit():
            return f"{code.zfill(6)}.{suffix}"
        if suffix == "HK" and code.isdigit():
            return f"{code.zfill(5)}.HK"
        raise DataVendorUnavailable(
            f"Tushare currently supports A-share, Hong Kong, and US tickers only, got '{symbol}'."
        )

    if raw.isdigit() and len(raw) <= 6:
        code = raw.zfill(6)
        if code.startswith(("6", "9", "5")):
            return f"{code}.SH"
        if code.startswith(("0", "2", "3")):
            return f"{code}.SZ"
        if code.startswith(("4", "8")):
            return f"{code}.BJ"
        return f"{raw.zfill(5)}.HK"

    if raw.replace("-", "").isalnum():
        return raw

    raise DataVendorUnavailable(
        f"Cannot map ticker '{symbol}' to a supported Tushare market automatically."
    )


@lru_cache(maxsize=1)
def _get_pro_client():
    token = (
        os.getenv("TUSHARE_TOKEN")
        or os.getenv("TUSHARE_API_TOKEN")
        or os.getenv("TS_TOKEN")
    )
    if not token:
        raise DataVendorUnavailable(
            "TUSHARE_TOKEN is not set. Configure token or use fallback vendor."
        )

    try:
        import tushare as ts
    except ImportError as exc:
        raise DataVendorUnavailable(
            "tushare package is not installed. Install it to enable tushare vendor."
        ) from exc

    try:
        ts.set_token(token)
        return ts.pro_api(token)
    except Exception as exc:
        raise DataVendorUnavailable(f"Failed to initialize tushare client: {exc}") from exc


def _to_csv_with_header(
    df: pd.DataFrame,
    title: str,
    summary_lines: list[str] | None = None,
) -> str:
    if df is None or df.empty:
        return f"No {title.lower()} data found."

    header = f"# {title}\n"
    header += f"# Total records: {len(df)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    if summary_lines:
        header += "# Key snapshot\n"
        header += "\n".join(summary_lines) + "\n\n"
    return header + df.to_csv(index=False)


def _to_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_billions(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number / 1e8:.2f}亿 CNY"


def _format_pct(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number:.2f}%"


def _format_multiple(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number:.2f}x"


def _format_price(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number:.2f} CNY/share"


def _format_market_value_10k_cny(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number / 1e4:.2f}亿 CNY"


def _append_if_present(
    lines: list[str],
    label: str,
    value,
    formatter: Callable | None = None,
):
    rendered = formatter(value) if formatter else value
    if rendered is None:
        return
    if not isinstance(rendered, str) and pd.isna(rendered):
        return
    if rendered == "":
        return
    lines.append(f"{label}: {rendered}")


def _safe_ratio(numerator, denominator) -> float | None:
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den


def _same_period_previous_year(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty or "end_date" not in df.columns:
        return None
    latest_end = str(df.iloc[0]["end_date"])
    if len(latest_end) != 8 or not latest_end.isdigit():
        return None
    prior_end = f"{int(latest_end[:4]) - 1}{latest_end[4:]}"
    prior_rows = df[df["end_date"].astype(str) == prior_end]
    if prior_rows.empty:
        return None
    return prior_rows.iloc[0]


def _trim_text(value, max_chars: int = 220) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text:
        return None
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _format_wan_to_billions(value) -> str | None:
    number = _to_float(value)
    if number is None:
        return None
    return f"{number / 1e4:.2f}亿 CNY"


def _prepare_latest_records(
    df: pd.DataFrame,
    cutoff_col: str | None = None,
    cutoff: str | None = None,
    sort_cols: tuple[str, ...] = (),
    dedupe_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    output = df.copy()
    if cutoff and cutoff_col and cutoff_col in output.columns:
        output = output[output[cutoff_col].astype(str) <= cutoff]
    if output.empty:
        return output

    sort_by: list[str] = []
    ascending: list[bool] = []
    for col in sort_cols:
        if col in output.columns:
            sort_by.append(col)
            ascending.append(False)
    if "update_flag" in output.columns:
        output = output.assign(
            _update_rank=pd.to_numeric(output["update_flag"], errors="coerce").fillna(0)
        )
        sort_by.append("_update_rank")
        ascending.append(False)

    if sort_by:
        output = output.sort_values(sort_by, ascending=ascending)
    else:
        output = output.sort_values(output.columns[0], ascending=False)

    subset = [col for col in dedupe_cols if col in output.columns]
    if subset:
        output = output.drop_duplicates(subset=subset, keep="first")

    if "_update_rank" in output.columns:
        output = output.drop(columns=["_update_rank"])
    return output


def _build_growth_and_valuation_snapshot(
    latest_price_row: pd.Series | None,
    fina_indicator_row: pd.Series | None,
) -> list[str]:
    if fina_indicator_row is None:
        return []

    lines: list[str] = []
    growth_specs = {
        "Revenue YoY": "or_yoy",
        "Net Profit YoY": "netprofit_yoy",
        "Deducted Net Profit YoY": "dt_netprofit_yoy",
        "Quarterly Revenue YoY": "q_sales_yoy",
        "Quarterly Operating Profit QoQ": "q_op_qoq",
    }
    for label, field in growth_specs.items():
        _append_if_present(lines, label, fina_indicator_row.get(field), _format_pct)

    pe_value = _to_float(latest_price_row.get("pe")) if latest_price_row is not None else None
    if pe_value is not None and pe_value > 0:
        for label, field in (
            ("Net Profit YoY", "netprofit_yoy"),
            ("Deducted Net Profit YoY", "dt_netprofit_yoy"),
            ("Revenue YoY", "or_yoy"),
        ):
            growth_value = _to_float(fina_indicator_row.get(field))
            if growth_value is not None and growth_value > 0:
                lines.append(f"PEG (using {label}): {pe_value / growth_value:.2f}x")
                break
    return lines


def _build_rd_snapshot(income_df: pd.DataFrame) -> list[str]:
    if income_df is None or income_df.empty:
        return []

    row = income_df.iloc[0]
    lines: list[str] = []
    _append_if_present(lines, "Latest Report Date", row.get("end_date"))
    _append_if_present(lines, "R&D Expense", row.get("rd_exp"), _format_billions)

    rd_exp = _to_float(row.get("rd_exp"))
    total_revenue = _to_float(row.get("total_revenue"))
    if rd_exp is not None and total_revenue not in (None, 0):
        lines.append(f"R&D Intensity: {rd_exp / total_revenue * 100:.2f}%")

    operate_profit = _to_float(row.get("operate_profit"))
    if rd_exp is not None and operate_profit not in (None, 0):
        lines.append(f"R&D / Operating Profit: {rd_exp / operate_profit * 100:.2f}%")

    prior_row = _same_period_previous_year(income_df)
    if prior_row is not None:
        prior_rd_exp = _to_float(prior_row.get("rd_exp"))
        if rd_exp is not None and prior_rd_exp not in (None, 0):
            lines.append(
                f"R&D Expense YoY (same period): {(rd_exp - prior_rd_exp) / prior_rd_exp * 100:.2f}%"
            )
    return lines


def _build_main_business_snapshot(
    company_df: pd.DataFrame | None,
    main_business_df: pd.DataFrame | None,
    end_api: str,
) -> list[str]:
    lines: list[str] = []

    if company_df is not None and not company_df.empty:
        row = company_df.iloc[0]
        _append_if_present(lines, "Main Business Summary", _trim_text(row.get("main_business"), 320))
        _append_if_present(lines, "Business Scope", _trim_text(row.get("business_scope"), 260))

    prepared = _prepare_latest_records(
        main_business_df,
        cutoff_col="end_date",
        cutoff=end_api,
        sort_cols=("end_date",),
        dedupe_cols=("end_date", "bz_item"),
    )
    if prepared.empty:
        return lines

    latest_end = str(prepared.iloc[0]["end_date"])
    latest = prepared[prepared["end_date"].astype(str) == latest_end].copy()
    if latest.empty or "bz_sales" not in latest.columns:
        return lines

    latest = latest[latest["bz_sales"].notna()].sort_values("bz_sales", ascending=False)
    if latest.empty:
        return lines

    disclosed_sales = pd.to_numeric(latest["bz_sales"], errors="coerce").fillna(0).sum()
    lines.append(f"Latest Segment Period: {latest_end}")
    for _, segment in latest.head(3).iterrows():
        pieces = [f"Segment: {segment.get('bz_item')}"]
        sales_value = _to_float(segment.get("bz_sales"))
        sales_text = _format_billions(sales_value)
        if sales_text is not None:
            pieces.append(f"Sales: {sales_text}")
        if sales_value is not None and disclosed_sales > 0:
            pieces.append(
                f"Share of disclosed segment sales: {sales_value / disclosed_sales * 100:.2f}%"
            )
        segment_margin = _safe_ratio(segment.get("bz_profit"), segment.get("bz_sales"))
        if segment_margin is not None:
            pieces.append(f"Segment Margin: {segment_margin * 100:.2f}%")
        lines.append(" | ".join(pieces))
    return lines


def _build_earnings_guidance_snapshot(
    forecast_df: pd.DataFrame | None,
    express_df: pd.DataFrame | None,
    end_api: str,
    latest_actual_end: str | None,
    total_market_value_10k: float | None,
) -> list[str]:
    lines: list[str] = []

    prepared_forecast = _prepare_latest_records(
        forecast_df,
        cutoff_col="ann_date",
        cutoff=end_api,
        sort_cols=("ann_date", "end_date", "first_ann_date"),
        dedupe_cols=("end_date",),
    )
    if not prepared_forecast.empty:
        row = prepared_forecast.iloc[0]
        _append_if_present(lines, "Latest Forecast Announcement Date", row.get("ann_date"))
        _append_if_present(lines, "Latest Forecast Period", row.get("end_date"))
        _append_if_present(lines, "Forecast Net Profit Min", row.get("net_profit_min"), _format_wan_to_billions)
        _append_if_present(lines, "Forecast Net Profit Max", row.get("net_profit_max"), _format_wan_to_billions)
        _append_if_present(lines, "Forecast Change Min", row.get("p_change_min"), _format_pct)
        _append_if_present(lines, "Forecast Change Max", row.get("p_change_max"), _format_pct)
        _append_if_present(lines, "Forecast Summary", _trim_text(row.get("summary"), 180))
        _append_if_present(lines, "Forecast Reason", _trim_text(row.get("change_reason"), 260))

        forecast_period = str(row.get("end_date")) if pd.notna(row.get("end_date")) else None
        forecast_min = _to_float(row.get("net_profit_min"))
        forecast_max = _to_float(row.get("net_profit_max"))
        forecast_midpoint = None
        if forecast_min is not None and forecast_max is not None:
            forecast_midpoint = (forecast_min + forecast_max) / 2
        elif forecast_min is not None:
            forecast_midpoint = forecast_min
        elif forecast_max is not None:
            forecast_midpoint = forecast_max

        if latest_actual_end and forecast_period and forecast_period > latest_actual_end:
            _append_if_present(
                lines,
                "Forecast Net Profit Midpoint",
                forecast_midpoint,
                _format_wan_to_billions,
            )
            if (
                total_market_value_10k is not None
                and total_market_value_10k > 0
                and forecast_midpoint is not None
                and forecast_midpoint > 0
            ):
                lines.append(
                    "Forward PE (market cap / forecast net profit midpoint): "
                    f"{total_market_value_10k / forecast_midpoint:.2f}x"
                )
        elif latest_actual_end:
            lines.append(
                "Forward PE Status: Latest available forecast is not newer than "
                f"the latest reported financial period {latest_actual_end}."
            )
        return lines

    prepared_express = _prepare_latest_records(
        express_df,
        cutoff_col="ann_date",
        cutoff=end_api,
        sort_cols=("ann_date", "end_date"),
        dedupe_cols=("end_date",),
    )
    if not prepared_express.empty:
        row = prepared_express.iloc[0]
        _append_if_present(lines, "Latest Earnings Express Period", row.get("end_date"))
        _append_if_present(lines, "Earnings Express Revenue", row.get("revenue"), _format_billions)
        _append_if_present(lines, "Earnings Express Net Income", row.get("n_income"), _format_billions)
        _append_if_present(lines, "Earnings Express Summary", _trim_text(row.get("perf_summary"), 180))

    if latest_actual_end:
        lines.append(
            "Forward PE Status: No current earnings guidance newer than "
            f"the latest reported financial period {latest_actual_end} was found."
        )
    return lines


def _extract_peer_keywords(company_df: pd.DataFrame | None) -> list[str]:
    if company_df is None or company_df.empty:
        return []
    row = company_df.iloc[0]
    text = " ".join(
        filter(
            None,
            [
                _trim_text(row.get("main_business"), 600),
                _trim_text(row.get("business_scope"), 600),
                _trim_text(row.get("introduction"), 600),
            ],
        )
    )
    if not text:
        return []

    strong_keywords = (
        "动力电池",
        "锂电池",
        "电池系统",
        "电池材料",
        "电池回收",
        "磷酸铁锂",
        "三元材料",
    )
    broad_keywords = ("储能",)

    matched_strong_keywords = [keyword for keyword in strong_keywords if keyword in text]
    if matched_strong_keywords:
        return matched_strong_keywords[:4]
    return [keyword for keyword in broad_keywords if keyword in text][:2]


def _load_keyword_peer_candidates(pro, keywords: list[str]) -> pd.DataFrame:
    if not keywords:
        return pd.DataFrame()

    company_frames = [
        pro.stock_company(exchange="SSE"),
        pro.stock_company(exchange="SZSE"),
        pro.stock_company(exchange="BSE"),
    ]
    companies = pd.concat(company_frames, ignore_index=True)
    business_text = (
        companies.get("main_business", pd.Series(dtype=object)).fillna("")
        + " "
        + companies.get("business_scope", pd.Series(dtype=object)).fillna("")
    )
    keyword_pattern = "|".join(re.escape(keyword) for keyword in keywords)
    matches = companies[business_text.str.contains(keyword_pattern, regex=True)]
    if matches.empty:
        return matches
    return matches[["ts_code"]].drop_duplicates()


def _build_peer_comparison_snapshot(
    pro,
    ts_code: str,
    industry: str | None,
    latest_trade_date: str | None,
    latest_price_row: pd.Series | None,
    fina_indicator_row: pd.Series | None,
    start_api_400d: str,
    end_api: str,
    company_df: pd.DataFrame | None = None,
) -> list[str]:
    if industry is None or pd.isna(industry) or latest_trade_date is None or pd.isna(latest_trade_date):
        return []

    peer_universe = pro.stock_basic(fields="ts_code,name,industry,market,list_status")
    if peer_universe is None or peer_universe.empty:
        return []

    peers = peer_universe[
        (peer_universe["industry"] == industry)
        & (peer_universe["list_status"] == "L")
        & (peer_universe["ts_code"] != ts_code)
    ]
    if peers.empty:
        return []

    peer_basis = f"same Tushare industry '{industry}'"
    keyword_candidates = _load_keyword_peer_candidates(pro, _extract_peer_keywords(company_df))
    if not keyword_candidates.empty:
        keyword_peers = peers.merge(keyword_candidates, on="ts_code", how="inner")
        if len(keyword_peers) >= 3:
            peers = keyword_peers
            keywords_display = ", ".join(_extract_peer_keywords(company_df))
            peer_basis = (
                f"same Tushare industry '{industry}' and business keywords [{keywords_display}]"
            )

    peer_valuation = pro.daily_basic(
        trade_date=latest_trade_date,
        fields="ts_code,close,pe,pb,ps,total_mv",
    )
    if peer_valuation is None or peer_valuation.empty:
        return []

    merged = peers.merge(peer_valuation, on="ts_code", how="inner")
    merged = merged[merged["total_mv"].notna()].sort_values("total_mv", ascending=False)
    sample = merged.head(3)
    if sample.empty:
        return []

    lines = [
        "Peer Sample Basis: "
        f"{peer_basis}, ranked by market value on {latest_trade_date}."
    ]
    peer_metrics: list[dict[str, float]] = []

    for _, peer in sample.iterrows():
        peer_indicator = _prepare_latest_records(
            pro.fina_indicator(ts_code=peer["ts_code"], start_date=start_api_400d, end_date=end_api),
            cutoff_col="end_date",
            cutoff=end_api,
            sort_cols=("end_date", "ann_date"),
            dedupe_cols=("end_date",),
        )
        peer_indicator_row = peer_indicator.iloc[0] if not peer_indicator.empty else None

        pieces = [f"{peer.get('name')} ({peer.get('ts_code')})"]
        _append_if_present(pieces, "Market Value", peer.get("total_mv"), _format_market_value_10k_cny)
        _append_if_present(pieces, "PE", peer.get("pe"), _format_multiple)
        _append_if_present(pieces, "PB", peer.get("pb"), _format_multiple)
        _append_if_present(pieces, "PS", peer.get("ps"), _format_multiple)
        if peer_indicator_row is not None:
            _append_if_present(pieces, "ROE", peer_indicator_row.get("roe"), _format_pct)
            _append_if_present(
                pieces,
                "Net Profit YoY",
                peer_indicator_row.get("netprofit_yoy"),
                _format_pct,
            )
        lines.append("Peer Sample: " + " | ".join(pieces))

        peer_metrics.append(
            {
                "pe": _to_float(peer.get("pe")),
                "pb": _to_float(peer.get("pb")),
                "ps": _to_float(peer.get("ps")),
                "roe": _to_float(peer_indicator_row.get("roe")) if peer_indicator_row is not None else None,
                "netprofit_yoy": (
                    _to_float(peer_indicator_row.get("netprofit_yoy"))
                    if peer_indicator_row is not None
                    else None
                ),
            }
        )

    comparisons: list[str] = []
    target_metric_map = {
        "PE": _to_float(latest_price_row.get("pe")) if latest_price_row is not None else None,
        "PB": _to_float(latest_price_row.get("pb")) if latest_price_row is not None else None,
        "PS": _to_float(latest_price_row.get("ps")) if latest_price_row is not None else None,
        "ROE": _to_float(fina_indicator_row.get("roe")) if fina_indicator_row is not None else None,
        "Net Profit YoY": (
            _to_float(fina_indicator_row.get("netprofit_yoy"))
            if fina_indicator_row is not None
            else None
        ),
    }
    peer_metric_map = {
        "PE": "pe",
        "PB": "pb",
        "PS": "ps",
        "ROE": "roe",
        "Net Profit YoY": "netprofit_yoy",
    }
    for label, metric_key in peer_metric_map.items():
        values = [item[metric_key] for item in peer_metrics if item.get(metric_key) is not None]
        target_value = target_metric_map[label]
        if not values or target_value is None:
            continue
        median_value = float(pd.Series(values).median())
        suffix = "%" if "YoY" in label or label == "ROE" else "x"
        comparisons.append(
            f"{label}: target {target_value:.2f}{suffix} vs sample median {median_value:.2f}{suffix}"
        )
    if comparisons:
        lines.append("Target vs Peer Median: " + " | ".join(comparisons))

    return lines


def _build_balance_sheet_summary(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    row = df.iloc[0]
    lines: list[str] = []
    _append_if_present(lines, "Latest Report Date", row.get("end_date"))
    _append_if_present(lines, "Total Assets", row.get("total_assets"), _format_billions)
    _append_if_present(lines, "Total Liabilities", row.get("total_liab"), _format_billions)
    _append_if_present(
        lines,
        "Equity Attributable to Shareholders",
        row.get("total_hldr_eqy_exc_min_int"),
        _format_billions,
    )
    debt_ratio = _safe_ratio(row.get("total_liab"), row.get("total_assets"))
    if debt_ratio is not None:
        lines.append(f"Asset-Liability Ratio: {debt_ratio * 100:.2f}%")
    _append_if_present(
        lines,
        "Current Assets",
        row.get("total_cur_assets"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Current Liabilities",
        row.get("total_cur_liab"),
        _format_billions,
    )
    current_ratio = _safe_ratio(row.get("total_cur_assets"), row.get("total_cur_liab"))
    if current_ratio is not None:
        lines.append(f"Current Ratio: {current_ratio:.2f}x")
    _append_if_present(lines, "Cash", row.get("money_cap"), _format_billions)
    _append_if_present(
        lines,
        "Accounts Receivable",
        row.get("accounts_receiv"),
        _format_billions,
    )
    _append_if_present(lines, "Inventories", row.get("inventories"), _format_billions)
    _append_if_present(
        lines,
        "Contract Liabilities",
        row.get("contract_liab"),
        _format_billions,
    )
    return lines


def _build_cashflow_summary(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    row = df.iloc[0]
    lines: list[str] = []
    _append_if_present(lines, "Latest Report Date", row.get("end_date"))
    _append_if_present(
        lines,
        "Operating Cash Flow",
        row.get("n_cashflow_act"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Investing Cash Flow",
        row.get("n_cashflow_inv_act"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Financing Cash Flow",
        row.get("n_cash_flows_fnc_act"),
        _format_billions,
    )
    free_cashflow = row.get("free_cashflow")
    if _to_float(free_cashflow) is None:
        operating_cash_flow = _to_float(row.get("n_cashflow_act"))
        capex_cash = _to_float(row.get("c_pay_acq_const_fiolta"))
        if operating_cash_flow is not None and capex_cash is not None:
            free_cashflow = operating_cash_flow - capex_cash
    _append_if_present(lines, "Free Cash Flow", free_cashflow, _format_billions)
    _append_if_present(
        lines,
        "Ending Cash and Cash Equivalents",
        row.get("c_cash_equ_end_period"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Beginning Cash and Cash Equivalents",
        row.get("c_cash_equ_beg_period"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Cash Received from Sales",
        row.get("c_fr_sale_sg"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Cash Paid for Goods and Services",
        row.get("c_paid_goods_s"),
        _format_billions,
    )
    _append_if_present(
        lines,
        "Capex Cash Outflow",
        row.get("c_pay_acq_const_fiolta"),
        _format_billions,
    )
    return lines


def _build_income_statement_summary(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    row = df.iloc[0]
    lines: list[str] = []
    _append_if_present(lines, "Latest Report Date", row.get("end_date"))
    _append_if_present(lines, "Total Revenue", row.get("total_revenue"), _format_billions)
    _append_if_present(
        lines,
        "Operating Profit",
        row.get("operate_profit"),
        _format_billions,
    )
    _append_if_present(lines, "Total Profit", row.get("total_profit"), _format_billions)
    _append_if_present(lines, "Net Income", row.get("n_income"), _format_billions)
    _append_if_present(
        lines,
        "Parent Net Income",
        row.get("n_income_attr_p"),
        _format_billions,
    )
    _append_if_present(lines, "R&D Expense", row.get("rd_exp"), _format_billions)
    _append_if_present(lines, "EBIT", row.get("ebit"), _format_billions)
    _append_if_present(lines, "EBITDA", row.get("ebitda"), _format_billions)
    rd_exp = _to_float(row.get("rd_exp"))
    total_revenue = _to_float(row.get("total_revenue"))
    if rd_exp is not None and total_revenue not in (None, 0):
        lines.append(f"R&D Intensity: {rd_exp / total_revenue * 100:.2f}%")
    operate_profit = _to_float(row.get("operate_profit"))
    if rd_exp is not None and operate_profit not in (None, 0):
        lines.append(f"R&D / Operating Profit: {rd_exp / operate_profit * 100:.2f}%")

    prior_row = _same_period_previous_year(df)
    if prior_row is not None:
        current_revenue = _to_float(row.get("total_revenue"))
        prior_revenue = _to_float(prior_row.get("total_revenue"))
        revenue_growth = None
        if current_revenue is not None and prior_revenue not in (None, 0):
            revenue_growth = (current_revenue - prior_revenue) / prior_revenue
        if revenue_growth is not None:
            lines.append(f"Revenue YoY (same period): {revenue_growth * 100:.2f}%")
        current_profit = _to_float(row.get("n_income_attr_p"))
        prior_profit = _to_float(prior_row.get("n_income_attr_p"))
        profit_growth = None
        if current_profit is not None and prior_profit not in (None, 0):
            profit_growth = (current_profit - prior_profit) / prior_profit
        if profit_growth is not None:
            lines.append(f"Parent Net Income YoY (same period): {profit_growth * 100:.2f}%")
        prior_rd_exp = _to_float(prior_row.get("rd_exp"))
        if rd_exp is not None and prior_rd_exp not in (None, 0):
            lines.append(f"R&D Expense YoY (same period): {(rd_exp - prior_rd_exp) / prior_rd_exp * 100:.2f}%")
    return lines


def _filter_statement(df: pd.DataFrame, freq: str, curr_date: str | None) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    output = df.copy()

    if curr_date and "end_date" in output.columns:
        cutoff = _to_api_date(curr_date)
        output = output[output["end_date"].astype(str) <= cutoff]

    if freq.lower() == "annual" and "end_date" in output.columns:
        output = output[output["end_date"].astype(str).str.endswith("1231")]

    sort_cols = []
    ascending = []
    if "end_date" in output.columns:
        sort_cols.append("end_date")
        ascending.append(False)
    if "update_flag" in output.columns:
        output = output.assign(
            _update_rank=pd.to_numeric(output["update_flag"], errors="coerce").fillna(0)
        )
        sort_cols.append("_update_rank")
        ascending.append(False)
    for col in ("ann_date", "f_ann_date"):
        if col in output.columns:
            sort_cols.append(col)
            ascending.append(False)

    sort_col = sort_cols[0] if sort_cols else output.columns[0]
    output = output.sort_values(sort_cols or sort_col, ascending=ascending or False)
    if "end_date" in output.columns:
        output = output.drop_duplicates(subset=["end_date"], keep="first")
    if "_update_rank" in output.columns:
        output = output.drop(columns=["_update_rank"])
    output = output.head(8)
    return output


def _fetch_price_data(pro, ts_code: str, start_api: str, end_api: str) -> pd.DataFrame:
    market = _classify_market(ts_code)
    if market == "a_share":
        return pro.daily(ts_code=ts_code, start_date=start_api, end_date=end_api)
    if market == "hk":
        return pro.hk_daily(ts_code=ts_code, start_date=start_api, end_date=end_api)
    return pro.us_daily(ts_code=ts_code, start_date=start_api, end_date=end_api)


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    pro = _get_pro_client()
    ts_code = _normalize_ts_code(symbol)

    start_api = _to_api_date(start_date)
    end_api = _to_api_date(end_date)

    data = _fetch_price_data(pro, ts_code, start_api, end_api)
    if data is None or data.empty:
        return f"No stock data found for '{ts_code}' between {start_date} and {end_date}."

    rename_map = {
        "trade_date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "vol": "Volume",
        "amount": "Amount",
        "pct_chg": "PctChg",
        "pre_close": "PrevClose",
        "change": "Change",
    }

    output = data.rename(columns=rename_map)
    if "Date" in output.columns:
        output["Date"] = pd.to_datetime(output["Date"], format="%Y%m%d").dt.strftime(
            "%Y-%m-%d"
        )
    output = output.sort_values("Date", ascending=True)

    preferred_cols = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "PrevClose",
        "Change",
        "PctChg",
        "Volume",
        "Amount",
    ]
    existing_cols = [c for c in preferred_cols if c in output.columns]
    output = output[existing_cols]

    return _to_csv_with_header(
        output,
        f"Tushare stock data for {ts_code} from {start_date} to {end_date}",
    )


def _load_price_frame(symbol: str, curr_date: str, look_back_days: int = 260) -> pd.DataFrame:
    pro = _get_pro_client()
    ts_code = _normalize_ts_code(symbol)
    end_dt = _parse_date(curr_date)
    start_dt = end_dt - timedelta(days=look_back_days)
    data = _fetch_price_data(
        pro,
        ts_code,
        start_dt.strftime("%Y%m%d"),
        end_dt.strftime("%Y%m%d"),
    )
    if data is None or data.empty:
        raise DataVendorUnavailable(
            f"No tushare price data found for '{ts_code}' before {curr_date}."
        )

    df = data.rename(
        columns={
            "trade_date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "vol": "Volume",
        }
    ).copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.sort_values("Date", ascending=True)
    return df[["Date", "Open", "High", "Low", "Close", "Volume"]]


def get_indicator(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int,
) -> str:
    descriptions = {
        "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
        "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
        "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
        "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
        "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
        "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
        "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
        "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
        "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
        "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.",
        "mfi": "MFI: Uses both price and volume to measure buying and selling pressure. Usage: Identify overbought (>80) or oversold (<20) conditions and confirm trends or reversals.",
    }
    if indicator not in descriptions:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(descriptions.keys())}"
        )

    current_dt = _parse_date(curr_date)
    start_dt = current_dt - timedelta(days=look_back_days)
    stats_df = wrap(_load_price_frame(symbol, curr_date))
    stats_df["Date"] = stats_df["Date"].dt.strftime("%Y-%m-%d")
    stats_df[indicator]

    lines = []
    probe_dt = current_dt
    while probe_dt >= start_dt:
        date_str = probe_dt.strftime("%Y-%m-%d")
        row = stats_df[stats_df["Date"] == date_str]
        if row.empty:
            lines.append(f"{date_str}: N/A: Not a trading day (weekend or holiday)")
        else:
            value = row.iloc[0][indicator]
            if pd.isna(value):
                lines.append(f"{date_str}: N/A")
            else:
                lines.append(f"{date_str}: {value}")
        probe_dt -= timedelta(days=1)

    return (
        f"## {indicator} values from {start_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + "\n".join(lines)
        + "\n\n"
        + descriptions[indicator]
    )


def get_fundamentals(ticker: str, curr_date: str | None = None) -> str:
    pro = _get_pro_client()
    ts_code = _normalize_ts_code(ticker)
    market = _classify_market(ts_code)

    if curr_date:
        curr_dt = _parse_date(curr_date)
    else:
        curr_dt = datetime.now()
        curr_date = curr_dt.strftime("%Y-%m-%d")

    end_api = curr_dt.strftime("%Y%m%d")
    start_api_40d = (curr_dt - timedelta(days=40)).strftime("%Y%m%d")
    start_api_400d = (curr_dt - timedelta(days=400)).strftime("%Y%m%d")
    stock_company = None
    main_business = None
    earnings_forecast = None
    earnings_express = None
    income_statement = None

    if market == "a_share":
        basic = pro.stock_basic(
            ts_code=ts_code,
            fields="ts_code,symbol,name,area,industry,market,list_date,list_status",
        )
        latest_price = pro.daily_basic(
            ts_code=ts_code,
            start_date=start_api_40d,
            end_date=end_api,
        )
        fina_indicator = pro.fina_indicator(
            ts_code=ts_code,
            start_date=start_api_400d,
            end_date=end_api,
        )
        stock_company = pro.stock_company(ts_code=ts_code)
        main_business = pro.fina_mainbz(ts_code=ts_code, type="P")
        earnings_forecast = pro.forecast(ts_code=ts_code)
        earnings_express = pro.express(ts_code=ts_code)
        income_statement = _filter_statement(pro.income(ts_code=ts_code), "quarterly", curr_date)
    elif market == "hk":
        basic = pro.hk_basic(ts_code=ts_code)
        latest_price = pro.hk_daily(ts_code=ts_code, start_date=start_api_40d, end_date=end_api)
        fina_indicator = None
    else:
        basic = pro.us_basic(ts_code=ts_code)
        latest_price = pro.us_daily(ts_code=ts_code, start_date=start_api_40d, end_date=end_api)
        fina_indicator = None

    overview_lines = [
        f"Ticker: {ts_code}",
        f"Market: {market}",
        f"Reference date: {curr_date}",
    ]
    company_profile_lines: list[str] = []
    valuation_lines: list[str] = []
    growth_lines: list[str] = []
    rd_lines: list[str] = []
    business_lines: list[str] = []
    guidance_lines: list[str] = []
    peer_lines: list[str] = []

    basic_row = None
    if basic is not None and not basic.empty:
        basic_row = basic.iloc[0]
        if market == "a_share":
            field_map = {
                "name": "Name",
                "area": "Area",
                "industry": "Industry",
                "market": "Market",
                "list_date": "List Date",
                "list_status": "List Status",
            }
        elif market == "hk":
            field_map = {
                "name": "Name",
                "fullname": "Full Name",
                "enname": "English Name",
                "market": "Market",
                "curr_type": "Currency",
                "list_date": "List Date",
                "list_status": "List Status",
            }
        else:
            field_map = {
                "name": "Name",
                "enname": "English Name",
                "classify": "Classify",
                "list_date": "List Date",
                "delist_date": "Delist Date",
            }
        for field, label in field_map.items():
            value = basic_row.get(field)
            if pd.notna(value):
                company_profile_lines.append(f"{label}: {value}")

    latest_price_row = None
    if latest_price is not None and not latest_price.empty:
        latest_price_row = latest_price.sort_values("trade_date", ascending=False).iloc[0]
        if market == "a_share":
            field_specs = {
                "trade_date": ("Latest Trade Date", None),
                "close": ("Latest Close Price", _format_price),
                "turnover_rate": ("Turnover Rate", _format_pct),
                "pe": ("PE", _format_multiple),
                "pb": ("PB", _format_multiple),
                "ps": ("PS", _format_multiple),
                "dv_ratio": ("Dividend Yield Ratio", _format_pct),
                "total_mv": ("Total Market Value", _format_market_value_10k_cny),
                "circ_mv": ("Circulating Market Value", _format_market_value_10k_cny),
            }
        else:
            field_specs = {
                "trade_date": ("Latest Trade Date", None),
                "close": ("Close", None),
                "open": ("Open", None),
                "high": ("High", None),
                "low": ("Low", None),
                "pre_close": ("Prev Close", None),
                "change": ("Change", None),
                "pct_chg": ("Pct Change", None),
                "vol": ("Volume", None),
                "amount": ("Amount", None),
            }
        for field, (label, formatter) in field_specs.items():
            _append_if_present(valuation_lines, label, latest_price_row.get(field), formatter)

    fina_indicator_row = None
    if fina_indicator is not None and not fina_indicator.empty:
        prepared_fina_indicator = _prepare_latest_records(
            fina_indicator,
            cutoff_col="end_date",
            cutoff=end_api,
            sort_cols=("end_date", "ann_date"),
            dedupe_cols=("end_date",),
        )
        fina_indicator_row = (
            prepared_fina_indicator.iloc[0] if not prepared_fina_indicator.empty else None
        )
        if fina_indicator_row is not None:
            field_specs = {
                "end_date": ("Latest Financial Period", None),
                "roe": ("ROE", _format_pct),
                "roa": ("ROA", _format_pct),
                "grossprofit_margin": ("Gross Margin", _format_pct),
                "netprofit_margin": ("Net Margin", _format_pct),
                "debt_to_assets": ("Debt to Assets", _format_pct),
                "ocf_to_or": ("OCF to Revenue", _format_pct),
            }
            for field, (label, formatter) in field_specs.items():
                _append_if_present(valuation_lines, label, fina_indicator_row.get(field), formatter)
        growth_lines.extend(_build_growth_and_valuation_snapshot(latest_price_row, fina_indicator_row))
    elif market == "hk":
        income = pro.hk_income(ts_code=ts_code, end_date=end_api)
        if income is not None and not income.empty:
            latest_end = income["end_date"].astype(str).max()
            valuation_lines.append(f"Latest Financial Period: {latest_end}")
            sample = income[income["end_date"].astype(str) == latest_end].head(12)
            for _, rec in sample.iterrows():
                valuation_lines.append(f"{rec.get('ind_name')}: {rec.get('ind_value')}")
    else:
        income = pro.us_income(ts_code=ts_code, end_date=end_api)
        if income is not None and not income.empty:
            latest_end = income["end_date"].astype(str).max()
            valuation_lines.append(f"Latest Financial Period: {latest_end}")
            sample = income[income["end_date"].astype(str) == latest_end].head(12)
            for _, rec in sample.iterrows():
                valuation_lines.append(f"{rec.get('ind_name')}: {rec.get('ind_value')}")

    if market == "a_share":
        if stock_company is not None and not stock_company.empty:
            stock_company_row = stock_company.iloc[0]
            _append_if_present(company_profile_lines, "Employees", stock_company_row.get("employees"))
            _append_if_present(
                company_profile_lines,
                "Company Introduction",
                _trim_text(stock_company_row.get("introduction"), 280),
            )

        rd_lines.extend(_build_rd_snapshot(income_statement))
        business_lines.extend(_build_main_business_snapshot(stock_company, main_business, end_api))

        latest_actual_end = None
        if fina_indicator_row is not None and pd.notna(fina_indicator_row.get("end_date")):
            latest_actual_end = str(fina_indicator_row.get("end_date"))
        elif income_statement is not None and not income_statement.empty:
            latest_actual_end = str(income_statement.iloc[0].get("end_date"))

        total_market_value_10k = (
            _to_float(latest_price_row.get("total_mv")) if latest_price_row is not None else None
        )
        guidance_lines.extend(
            _build_earnings_guidance_snapshot(
                earnings_forecast,
                earnings_express,
                end_api,
                latest_actual_end,
                total_market_value_10k,
            )
        )
        peer_lines.extend(
            _build_peer_comparison_snapshot(
                pro,
                ts_code,
                basic_row.get("industry") if basic_row is not None else None,
                str(latest_price_row.get("trade_date")) if latest_price_row is not None else None,
                latest_price_row,
                fina_indicator_row,
                start_api_400d,
                end_api,
                stock_company,
            )
        )

    header = f"# Tushare fundamentals for {ts_code}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    sections = []
    for title, section_lines in (
        ("Overview", overview_lines),
        ("Company Profile", company_profile_lines),
        ("Valuation and Profitability Snapshot", valuation_lines),
        ("Growth and PEG Snapshot", growth_lines),
        ("R&D Snapshot", rd_lines),
        ("Main Business and Segment Mix", business_lines),
        ("Earnings Guidance and Forward Valuation", guidance_lines),
        ("Peer Comparison Snapshot", peer_lines),
    ):
        if section_lines:
            sections.append(f"## {title}\n" + "\n".join(section_lines))

    return header + "\n\n".join(sections)


def _statement_common(
    ticker: str,
    freq: str,
    curr_date: str | None,
    fetcher: Callable,
    title: str,
    summary_builder: Callable[[pd.DataFrame], list[str]] | None = None,
) -> str:
    pro = _get_pro_client()
    ts_code = _normalize_ts_code(ticker)
    market = _classify_market(ts_code)
    data = fetcher(pro, ts_code, market)
    filtered = _filter_statement(data, freq, curr_date)
    return _to_csv_with_header(
        filtered,
        f"Tushare {title} for {ts_code} ({freq})",
        summary_builder(filtered) if summary_builder else None,
    )


def get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _statement_common(
        ticker,
        freq,
        curr_date,
        lambda pro, ts_code, market: (
            pro.balancesheet(ts_code=ts_code)
            if market == "a_share"
            else pro.hk_balancesheet(ts_code=ts_code)
            if market == "hk"
            else pro.us_balancesheet(ts_code=ts_code)
        ),
        "balance sheet",
        _build_balance_sheet_summary,
    )


def get_cashflow(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _statement_common(
        ticker,
        freq,
        curr_date,
        lambda pro, ts_code, market: (
            pro.cashflow(ts_code=ts_code)
            if market == "a_share"
            else pro.hk_cashflow(ts_code=ts_code)
            if market == "hk"
            else pro.us_cashflow(ts_code=ts_code)
        ),
        "cashflow",
        _build_cashflow_summary,
    )


def get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: str | None = None,
) -> str:
    return _statement_common(
        ticker,
        freq,
        curr_date,
        lambda pro, ts_code, market: (
            pro.income(ts_code=ts_code)
            if market == "a_share"
            else pro.hk_income(ts_code=ts_code)
            if market == "hk"
            else pro.us_income(ts_code=ts_code)
        ),
        "income statement",
        _build_income_statement_summary,
    )


def get_insider_transactions(ticker: str) -> str:
    pro = _get_pro_client()
    ts_code = _normalize_ts_code(ticker)
    market = _classify_market(ts_code)

    if market != "a_share":
        raise DataVendorUnavailable(
            f"Tushare insider transactions currently support A-share tickers only, got '{ts_code}'."
        )

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=365)

    try:
        data = pro.stk_holdertrade(
            ts_code=ts_code,
            start_date=start_dt.strftime("%Y%m%d"),
            end_date=end_dt.strftime("%Y%m%d"),
        )
    except Exception as exc:
        raise DataVendorUnavailable(
            f"Failed to retrieve tushare insider transactions for '{ts_code}': {exc}"
        ) from exc

    if data is None or data.empty:
        return f"No tushare insider transactions found for '{ts_code}'."

    output = data.rename(
        columns={
            "ann_date": "AnnouncementDate",
            "holder_name": "HolderName",
            "holder_type": "HolderType",
            "in_de": "Direction",
            "change_vol": "ChangeVolume",
            "change_ratio": "ChangeRatio",
            "after_share": "AfterShareholding",
            "after_ratio": "AfterRatio",
            "avg_price": "AveragePrice",
            "total_share": "TotalShareholding",
            "begin_date": "StartDate",
            "close_date": "EndDate",
        }
    ).copy()

    for col in ("AnnouncementDate", "StartDate", "EndDate"):
        if col in output.columns:
            output[col] = pd.to_datetime(
                output[col], format="%Y%m%d", errors="coerce"
            ).dt.strftime("%Y-%m-%d")

    preferred_cols = [
        "AnnouncementDate",
        "HolderName",
        "HolderType",
        "Direction",
        "ChangeVolume",
        "ChangeRatio",
        "AfterShareholding",
        "AfterRatio",
        "AveragePrice",
        "TotalShareholding",
        "StartDate",
        "EndDate",
    ]
    existing_cols = [col for col in preferred_cols if col in output.columns]
    if existing_cols:
        output = output[existing_cols]

    sort_col = "AnnouncementDate" if "AnnouncementDate" in output.columns else output.columns[0]
    output = output.sort_values(sort_col, ascending=False)
    return _to_csv_with_header(output, f"Tushare insider transactions for {ts_code}")
