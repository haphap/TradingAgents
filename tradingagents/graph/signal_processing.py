# TradingAgents/graph/signal_processing.py

import re

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        normalized = self._normalize_known_rating(full_signal)
        if normalized:
            return normalized

        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "The report may express the final recommendation in English or Chinese. "
                "Map Chinese ratings as follows: 买入=BUY, 增持=OVERWEIGHT, 持有=HOLD, 减持=UNDERWEIGHT, 卖出=SELL. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content

    @staticmethod
    def _normalize_known_rating(full_signal: str) -> str | None:
        text = (full_signal or "").upper()
        english_markers = {
            "FINAL TRANSACTION PROPOSAL: **BUY**": "BUY",
            "FINAL TRANSACTION PROPOSAL: **OVERWEIGHT**": "OVERWEIGHT",
            "FINAL TRANSACTION PROPOSAL: **HOLD**": "HOLD",
            "FINAL TRANSACTION PROPOSAL: **UNDERWEIGHT**": "UNDERWEIGHT",
            "FINAL TRANSACTION PROPOSAL: **SELL**": "SELL",
        }
        for marker, rating in english_markers.items():
            if marker in text:
                return rating

        chinese_markers = {
            "最终交易建议: **买入**": "BUY",
            "最终交易建议: **增持**": "OVERWEIGHT",
            "最终交易建议: **持有**": "HOLD",
            "最终交易建议: **减持**": "UNDERWEIGHT",
            "最终交易建议: **卖出**": "SELL",
        }
        for marker, rating in chinese_markers.items():
            if marker in full_signal:
                return rating

        for pattern, rating in (
            (r"(?:RATING|RECOMMENDATION)\s*[:：]\s*\**\s*BUY\b", "BUY"),
            (r"(?:RATING|RECOMMENDATION)\s*[:：]\s*\**\s*OVERWEIGHT\b", "OVERWEIGHT"),
            (r"(?:RATING|RECOMMENDATION)\s*[:：]\s*\**\s*HOLD\b", "HOLD"),
            (r"(?:RATING|RECOMMENDATION)\s*[:：]\s*\**\s*UNDERWEIGHT\b", "UNDERWEIGHT"),
            (r"(?:RATING|RECOMMENDATION)\s*[:：]\s*\**\s*SELL\b", "SELL"),
        ):
            if re.search(pattern, text):
                return rating

        for pattern, rating in (
            (r"(?:评级|建议)\s*[:：]\s*\**\s*买入", "BUY"),
            (r"(?:评级|建议)\s*[:：]\s*\**\s*增持", "OVERWEIGHT"),
            (r"(?:评级|建议)\s*[:：]\s*\**\s*持有", "HOLD"),
            (r"(?:评级|建议)\s*[:：]\s*\**\s*减持", "UNDERWEIGHT"),
            (r"(?:评级|建议)\s*[:：]\s*\**\s*卖出", "SELL"),
        ):
            if re.search(pattern, full_signal):
                return rating

        return None
