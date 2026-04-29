"""End-to-end smoke for structured-output agents against a real provider.

This keeps the surface tight: it exercises the Research Manager, Trader,
Portfolio Manager, and the deterministic SignalProcessor adapter.
"""

from __future__ import annotations

import argparse
import sys

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.trader.trader import create_trader
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.llm_clients import create_llm_client


PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "deepseek": "deepseek-chat",
    "glm": "glm-5",
    "google": "gemini-2.5-flash",
    "openai": "gpt-5.4-mini",
    "qwen": "qwen-plus",
    "xai": "grok-4",
}

DEBATE_HISTORY = """
Bull Analyst: NVDA's data-center revenue grew 60% YoY last quarter, driven by
Blackwell ramp and sovereign AI deals. Margins remain above peer average.

Bear Analyst: Concentration risk is real — top three customers are >40% of
revenue. Any pause in hyperscaler capex would compress the multiple.
"""


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-20",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_debate_state": {
            "history": DEBATE_HISTORY,
            "bull_history": "Bull Analyst: NVDA's data-center revenue grew 60% YoY...",
            "bear_history": "Bear Analyst: Concentration risk is real...",
            "current_response": "",
            "current_bull_response": "",
            "current_bear_response": "",
            "bull_snapshot": "",
            "bear_snapshot": "",
            "bull_snapshot_path": "",
            "bear_snapshot_path": "",
            "debate_brief": "",
            "latest_speaker": "",
            "judge_decision": "",
            "judge_snapshot": "",
            "judge_snapshot_path": "",
            "count": 1,
        },
    }


def _make_trader_state(investment_plan: str):
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-20",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_plan": investment_plan,
    }


def _make_pm_state(investment_plan: str, trader_plan: str):
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-04-20",
        "past_context": "",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_plan": investment_plan,
        "trader_investment_plan": trader_plan,
        "risk_debate_state": {
            "history": "Aggressive: lean in. Conservative: trim. Neutral: balanced sizing.",
            "aggressive_history": "Aggressive: ...",
            "conservative_history": "Conservative: ...",
            "neutral_history": "Neutral: ...",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "aggressive_snapshot": "",
            "conservative_snapshot": "",
            "neutral_snapshot": "",
            "aggressive_snapshot_path": "",
            "conservative_snapshot_path": "",
            "neutral_snapshot_path": "",
            "debate_brief": "",
            "judge_decision": "",
            "judge_snapshot": "",
            "judge_snapshot_path": "",
            "count": 1,
        },
    }


def _print_section(title: str, content: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}\n{content}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=sorted(PROVIDER_DEFAULTS))
    parser.add_argument("--deep-model", default=None, help="Override deep_think_llm")
    parser.add_argument("--quick-model", default=None, help="Override quick_think_llm")
    args = parser.parse_args()

    default_model = PROVIDER_DEFAULTS[args.provider]
    deep_model = args.deep_model or default_model
    quick_model = args.quick_model or default_model

    deep_llm = create_llm_client(provider=args.provider, model=deep_model).get_llm()
    quick_llm = create_llm_client(provider=args.provider, model=quick_model).get_llm()

    research_manager = create_research_manager(deep_llm)
    research_result = research_manager(_make_rm_state())
    investment_plan = research_result["investment_plan"]
    _print_section("[1] Research Manager — investment_plan", investment_plan)

    trader = create_trader(quick_llm)
    trader_result = trader(_make_trader_state(investment_plan))
    trader_plan = trader_result["trader_investment_plan"]
    _print_section("[2] Trader — trader_investment_plan", trader_plan)

    portfolio_manager = create_portfolio_manager(deep_llm)
    portfolio_result = portfolio_manager(_make_pm_state(investment_plan, trader_plan))
    final_decision = portfolio_result["final_trade_decision"]
    _print_section("[3] Portfolio Manager — final_trade_decision", final_decision)

    rating = SignalProcessor().process_signal(final_decision)
    _print_section("[4] SignalProcessor → rating", rating)

    checks = [
        ("Research Manager", investment_plan, ("## ",)),
        ("Trader", trader_plan, ("FINAL TRANSACTION PROPOSAL:", "最终交易建议:")),
        ("Portfolio Manager", final_decision, ("FEEDBACK SNAPSHOT:", "反馈快照:")),
    ]
    failures = 0
    print("\n" + "=" * 70 + "\nStructure checks\n" + "=" * 70)
    for name, text, alternatives in checks:
        ok = any(marker in text for marker in alternatives)
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: contains one of {alternatives!r}")
        failures += int(not ok)

    if failures:
        print(f"\nSmoke FAILED: {failures} structure check(s) missing.")
        return 1
    print("\nSmoke PASSED: structured-output flow is working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
