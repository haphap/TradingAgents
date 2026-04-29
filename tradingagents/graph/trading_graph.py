import logging
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

import yfinance as yf

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100)
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor()

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""
        init_agent_state, args, _ = self.prepare_run(company_name, trade_date)

        try:
            if self.debug:
                trace = []
                for chunk in self.graph.stream(init_agent_state, **args):
                    if chunk["messages"]:
                        chunk["messages"][-1].pretty_print()
                    trace.append(chunk)

                final_state = trace[-1]
            else:
                final_state = self.graph.invoke(init_agent_state, **args)

            self.finalize_run(trade_date, final_state)
            return final_state, self.process_signal(final_state["final_trade_decision"])
        finally:
            self.close_run()

    def prepare_run(
        self,
        company_name: str,
        trade_date: str,
        callbacks: Optional[List] = None,
    ) -> tuple[Dict[str, Any] | None, Dict[str, Any], bool]:
        """Prepare graph execution and return (input_state, graph_args, resumed)."""
        self.close_run()
        self.ticker = company_name
        trade_date_str = str(trade_date)
        resumed = False
        self._resolve_pending_entries(company_name)

        if self.config.get("checkpoint_enabled"):
            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, trade_date_str
            )
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)
            resumed = step is not None
            if resumed:
                logger.info(
                    "Resuming from step %d for %s on %s",
                    step,
                    company_name,
                    trade_date_str,
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date_str)

        init_agent_state = None
        if not resumed:
            init_agent_state = self.propagator.create_initial_state(
                company_name,
                trade_date,
                past_context=self.memory_log.get_past_context(company_name),
            )

        args = self.propagator.get_graph_args(callbacks=callbacks)
        if self.config.get("checkpoint_enabled"):
            args.setdefault("config", {}).setdefault("configurable", {})[
                "thread_id"
            ] = thread_id(company_name, trade_date_str)

        return init_agent_state, args, resumed

    def finalize_run(self, trade_date, final_state: Dict[str, Any]) -> None:
        """Persist a successful run and clear its checkpoint."""
        self.curr_state = final_state
        self._log_state(trade_date, final_state)
        self.memory_log.store_decision(
            final_state["company_of_interest"],
            str(trade_date),
            final_state["final_trade_decision"],
        )

        if self.config.get("checkpoint_enabled") and self.ticker:
            clear_checkpoint(
                self.config["data_cache_dir"], self.ticker, str(trade_date)
            )

    def close_run(self) -> None:
        """Close any active checkpoint context and restore the default graph."""
        if self._checkpointer_ctx is None:
            return

        self._checkpointer_ctx.__exit__(None, None, None)
        self._checkpointer_ctx = None
        self.graph = self.workflow.compile()

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def _fetch_returns(self, ticker: str, trade_date: str, holding_days: int = 5):
        """Fetch stock and benchmark returns after a trade date."""
        start = datetime.strptime(str(trade_date), "%Y-%m-%d")
        end = start + timedelta(days=holding_days + 7)

        stock_history = yf.Ticker(ticker).history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )
        benchmark_history = yf.Ticker("SPY").history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
        )

        if len(stock_history) < 2 or len(benchmark_history) < 2:
            return None, None, None

        end_idx = min(holding_days, len(stock_history) - 1, len(benchmark_history) - 1)
        stock_raw = (
            float(stock_history["Close"].iloc[end_idx]) / float(stock_history["Close"].iloc[0])
        ) - 1.0
        benchmark_raw = (
            float(benchmark_history["Close"].iloc[end_idx]) / float(benchmark_history["Close"].iloc[0])
        ) - 1.0
        return stock_raw, stock_raw - benchmark_raw, end_idx

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve any past pending entries for this ticker when returns are now available."""
        updates = []
        for entry in self.memory_log.get_pending_entries():
            if entry["ticker"] != ticker:
                continue
            raw_return, alpha_return, holding_days = self._fetch_returns(
                entry["ticker"], entry["date"]
            )
            if raw_return is None or alpha_return is None or holding_days is None:
                continue
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry["decision"],
                raw_return=raw_return,
                alpha_return=alpha_return,
            )
            updates.append(
                {
                    "ticker": entry["ticker"],
                    "trade_date": entry["date"],
                    "raw_return": raw_return,
                    "alpha_return": alpha_return,
                    "holding_days": holding_days,
                    "reflection": reflection,
                }
            )

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def reflect_and_remember(self, returns_losses):
        raise RuntimeError(
            "Deferred reflections are automatic now. Re-run the ticker after return data is available to resolve pending memory-log entries."
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
