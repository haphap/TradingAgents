from enum import Enum

from pydantic import BaseModel, Field

from tradingagents.agents.utils.agent_utils import (
    get_output_language,
    localize_label,
    localize_rating_term,
)


class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class ResearchPlan(BaseModel):
    debate_conclusion: str = Field(description="Balanced conclusion after evaluating both bull and bear cases.")
    action_logic: str = Field(description="Evidence-to-action logic explaining valuation, catalysts, risks, and triggers.")
    positioning_recommendation: str = Field(description="Actionable trading guidance with execution details.")
    rating: PortfolioRating = Field(description="Final research-manager rating.")
    snapshot_stance: str = Field(description="Concise stance for the feedback snapshot.")
    snapshot_new_and_rebuttal: str = Field(description="What was newly added this round and how it rebuts the opposing case.")
    snapshot_to_verify: str = Field(description="Specific follow-up points or triggers to verify next.")


class TraderProposal(BaseModel):
    thesis: str = Field(description="Concise trading thesis explaining the proposed action.")
    execution_plan: str = Field(description="Concrete execution plan with entry, add, reduce, or exit conditions.")
    risk_management: str = Field(description="Risk controls, invalidation signals, and monitoring items.")
    rating: PortfolioRating = Field(description="Trader recommendation.")


class PortfolioDecision(BaseModel):
    debate_conclusion: str = Field(description="Synthesis of the full risk debate across all perspectives.")
    action_logic: str = Field(description="Portfolio manager logic from evidence to sizing and execution.")
    positioning_recommendation: str = Field(description="Final actionable portfolio recommendation and implementation guidance.")
    rating: PortfolioRating = Field(description="Final portfolio-manager rating.")
    snapshot_stance: str = Field(description="Concise stance for the feedback snapshot.")
    snapshot_new_and_rebuttal: str = Field(description="What was newly added this round and how it rebutted competing views.")
    snapshot_to_verify: str = Field(description="Specific items or triggers to verify next.")


def _is_chinese_output() -> bool:
    return get_output_language().strip().lower() in {"chinese", "中文", "zh", "zh-cn", "zh-hans"}


def _render_snapshot(stance: str, new_and_rebuttal: str, to_verify: str) -> str:
    if _is_chinese_output():
        return (
            "反馈快照:\n"
            f"- 立场: {stance.strip()}\n"
            f"- 本轮新增与反驳: {new_and_rebuttal.strip()}\n"
            f"- 待验证: {to_verify.strip()}"
        )
    return (
        "FEEDBACK SNAPSHOT:\n"
        f"- Stance: {stance.strip()}\n"
        f"- New this round & rebuttal: {new_and_rebuttal.strip()}\n"
        f"- To verify: {to_verify.strip()}"
    )


def render_research_plan(plan: ResearchPlan) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    body = (
        f"## {localize_label('Debate Conclusion', '辩论结论')}\n"
        f"{plan.debate_conclusion.strip()}\n\n"
        f"## {localize_label('Action Logic', '行为逻辑')}\n"
        f"{plan.action_logic.strip()}\n\n"
        f"## {localize_label('Positioning Recommendation', '持仓建议')}\n"
        f"{localize_label('Recommendation', '建议评级')}: {recommendation}\n"
        f"{plan.positioning_recommendation.strip()}"
    )
    snapshot = _render_snapshot(
        plan.snapshot_stance or recommendation,
        plan.snapshot_new_and_rebuttal,
        plan.snapshot_to_verify,
    )
    return f"{body}\n\n{snapshot}".strip()


def render_trader_proposal(plan: TraderProposal) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    if _is_chinese_output():
        return (
            "## 交易逻辑\n"
            f"{plan.thesis.strip()}\n\n"
            "## 执行计划\n"
            f"{plan.execution_plan.strip()}\n\n"
            "## 风险控制\n"
            f"{plan.risk_management.strip()}\n\n"
            f"最终交易建议: **{recommendation}**"
        ).strip()
    return (
        "## Trading Thesis\n"
        f"{plan.thesis.strip()}\n\n"
        "## Execution Plan\n"
        f"{plan.execution_plan.strip()}\n\n"
        "## Risk Management\n"
        f"{plan.risk_management.strip()}\n\n"
        f"FINAL TRANSACTION PROPOSAL: **{recommendation.upper()}**"
    ).strip()


def render_portfolio_decision(plan: PortfolioDecision) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    if _is_chinese_output():
        final_line = f"最终交易建议: **{recommendation}**"
    else:
        final_line = f"FINAL TRANSACTION PROPOSAL: **{recommendation.upper()}**"

    body = (
        f"## {localize_label('Debate Conclusion', '辩论结论')}\n"
        f"{plan.debate_conclusion.strip()}\n\n"
        f"## {localize_label('Action Logic', '行为逻辑')}\n"
        f"{plan.action_logic.strip()}\n\n"
        f"## {localize_label('Positioning Recommendation', '持仓建议')}\n"
        f"{localize_label('Recommendation', '建议评级')}: {recommendation}\n"
        f"{plan.positioning_recommendation.strip()}\n\n"
        f"{final_line}"
    )
    snapshot = _render_snapshot(
        plan.snapshot_stance or recommendation,
        plan.snapshot_new_and_rebuttal,
        plan.snapshot_to_verify,
    )
    return f"{body}\n\n{snapshot}".strip()
