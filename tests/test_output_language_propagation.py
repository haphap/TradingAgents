import copy
import unittest

from cli.main import format_research_team_history, format_risk_management_history
from tradingagents.agents.schemas import (
    PortfolioDecision,
    PortfolioRating,
    ResearchPlan,
    TraderProposal,
)
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.agent_utils import (
    get_collaboration_stop_instruction,
    normalize_chinese_role_terms,
)
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.default_config import DEFAULT_CONFIG


class _FakeResponse:
    def __init__(self, content="ok"):
        self.content = content


class _CapturingLLM:
    def __init__(self):
        self.calls = []
        self.structured_calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        return _FakeResponse(
            "测试输出\n"
            "决策摘要:\n"
            "- 评级: 持有\n"
            "- 置信度: 60%\n"
            "- 时间区间: 1-3个月\n"
            "- 关键假设:\n"
            "  1. 需求平稳。\n"
            "  2. 波动可控。\n"
            "  3. 等待验证。\n"
            "反馈快照:\n"
            "- 当前观点: x\n"
            "- 发生了什么变化: y\n"
            "- 为什么变化: z\n"
            "- 关键反驳: r\n"
            "- 下一轮教训: l"
        )

    def with_structured_output(self, schema):
        parent = self

        class _StructuredInvoker:
            def invoke(self, prompt):
                parent.structured_calls.append(prompt)
                if schema is TraderProposal:
                    return TraderProposal(
                        thesis="交易逻辑测试。",
                        execution_plan="1. 等待确认后分批执行。",
                        risk_management="跌破关键位就减仓，并继续跟踪成交量。",
                        rating=PortfolioRating.HOLD,
                    )
                if schema is ResearchPlan:
                    return ResearchPlan(
                        debate_conclusion="多空双方都提出了有效证据，但多头证据略强。",
                        action_logic="估值仍需消化，但催化节奏与盈利兑现共同支持保留上行敞口。",
                        positioning_recommendation="维持增持，等待确认后分批加仓，并跟踪风险边界。",
                        rating=PortfolioRating.OVERWEIGHT,
                        snapshot_stance="增持",
                        snapshot_new_and_rebuttal="本轮补充了盈利兑现与估值约束之间的平衡关系。",
                        snapshot_to_verify="继续跟踪订单、毛利率与资本开支。",
                    )
                if schema is PortfolioDecision:
                    return PortfolioDecision(
                        debate_conclusion="保守与中性观点限制了仓位，但激进观点提供了上行线索。",
                        action_logic="在估值、催化节奏与仓位约束之间平衡后，当前更适合维持中性偏积极配置。",
                        positioning_recommendation="先保留基础仓位，确认催化后再加仓，并设置回撤风控。",
                        rating=PortfolioRating.HOLD,
                        snapshot_stance="持有",
                        snapshot_new_and_rebuttal="新增了对仓位节奏与风险预算的约束。",
                        snapshot_to_verify="继续跟踪波动率、资金流和业绩兑现。",
                    )
                raise AssertionError(f"Unexpected schema: {schema}")

        return _StructuredInvoker()


class _EmptyMemory:
    def get_memories(self, *_args, **_kwargs):
        return []


class _MemoryWithLessons:
    def get_memories(self, *_args, **_kwargs):
        return [{"recommendation": "Keep sizing small until demand confirms."}]


class OutputLanguagePropagationTests(unittest.TestCase):
    def setUp(self):
        self.original_config = copy.deepcopy(get_config())
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["output_language"] = "Chinese"
        set_config(cfg)

        self.base_state = {
            "company_of_interest": "002155.SZ",
            "trade_date": "2026-04-28",
            "past_context": "",
            "investment_plan": "Plan",
            "market_report": "Market report",
            "sentiment_report": "Sentiment report",
            "news_report": "News report",
            "fundamentals_report": "Fundamentals report",
            "trader_investment_plan": "Trader plan",
            "risk_debate_state": {
                "history": "",
                "aggressive_history": "",
                "conservative_history": "",
                "neutral_history": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "aggressive_snapshot": "",
                "conservative_snapshot": "",
                "neutral_snapshot": "",
                "debate_brief": "",
                "judge_decision": "",
                "judge_snapshot": "",
                "judge_snapshot_path": "",
                "count": 0,
            },
            "investment_debate_state": {
                "history": "",
                "bear_history": "",
                "bull_history": "",
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
                "count": 0,
            },
        }

    def tearDown(self):
        set_config(self.original_config)

    def test_trader_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_trader(llm, _EmptyMemory())
        node(self.base_state)

        system_prompt = llm.structured_calls[0][0]["content"]
        self.assertIn("Write your entire response in Chinese.", system_prompt)
        self.assertIn("时机", system_prompt)

    def test_research_manager_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_research_manager(llm, _EmptyMemory())
        node(self.base_state)

        prompt = llm.structured_calls[0]
        self.assertIn("Write your entire response in Chinese.", prompt)
        self.assertIn("多头分析师", prompt)
        self.assertIn("空头分析师", prompt)
        self.assertIn("买入", prompt)
        self.assertIn("持有", prompt)
        self.assertIn("催化节奏", prompt)
        self.assertNotIn("catalyst timing", prompt)

    def test_bull_bear_researcher_prompts_require_chinese_body_and_decision_summary(self):
        for factory in (create_bull_researcher, create_bear_researcher):
            llm = _CapturingLLM()
            node = factory(llm, _EmptyMemory())
            node(self.base_state)

            prompt = llm.calls[0]
            self.assertIn("written entirely in Chinese", prompt)
            self.assertIn("决策摘要", prompt)
            self.assertNotIn("Internal lessons from similar situations", prompt)
            self.assertIn("Write your entire response in Chinese.", prompt)
            self.assertIn("Do not use variants like", prompt)
            self.assertIn("牛派分析师", prompt)
            self.assertIn("熊派分析师", prompt)
            self.assertIn("反馈快照", prompt)
            self.assertIn("关键约束", prompt)

    def test_portfolio_manager_prompt_includes_past_context_only_when_present(self):
        llm = _CapturingLLM()
        state = copy.deepcopy(self.base_state)
        state["past_context"] = "Past analyses of 002155.SZ:\n[2026-01-10 | 002155.SZ | Hold | +2.0% | +0.5% | 5d]"
        node = create_portfolio_manager(llm, _EmptyMemory())
        node(state)

        prompt = llm.structured_calls[0]
        self.assertIn("历史决策复盘", prompt)
        self.assertIn("Past analyses of 002155.SZ", prompt)

    def test_research_team_history_keeps_real_snapshot_blocks(self):
        llm = _CapturingLLM()
        node = create_bull_researcher(llm, _EmptyMemory())

        investment_debate_state = node(copy.deepcopy(self.base_state))["investment_debate_state"]
        formatted = format_research_team_history(investment_debate_state)

        self.assertIn("决策摘要:", formatted)
        self.assertIn("- 评级: 持有", formatted)
        self.assertNotIn("##### 本轮复盘", formatted)
        self.assertNotIn("##### 自动复盘", formatted)
        self.assertIn("- 立场: x", formatted)
        self.assertIn("- 本轮新增与反驳: y；z；r", formatted)
        self.assertNotIn("决策摘要", investment_debate_state["current_bull_response"])
        self.assertNotIn("反馈快照", investment_debate_state["current_bull_response"])

    def test_normalize_chinese_role_terms_replaces_display_variants(self):
        text = "我是熊派分析师，也不同意牛派分析师、激进分析师、保守分析师、中性分析师、熊派投资者和根本分析的说法。"
        normalized = normalize_chinese_role_terms(text)

        self.assertNotIn("熊派分析师", normalized)
        self.assertNotIn("牛派分析师", normalized)
        self.assertNotIn("激进分析师", normalized)
        self.assertNotIn("保守分析师", normalized)
        self.assertNotIn("中性分析师", normalized)
        self.assertNotIn("熊派投资者", normalized)
        self.assertNotIn("根本分析", normalized)
        self.assertIn("空头分析师", normalized)
        self.assertIn("多头分析师", normalized)
        self.assertIn("激进风险分析师", normalized)
        self.assertIn("保守风险分析师", normalized)
        self.assertIn("中性风险分析师", normalized)
        self.assertIn("空头投资者", normalized)
        self.assertIn("基本面分析", normalized)

    def test_risk_team_prompts_respect_output_language(self):
        for factory in (
            create_aggressive_debator,
            create_conservative_debator,
            create_neutral_debator,
        ):
            llm = _CapturingLLM()
            node = factory(llm)
            node(self.base_state)

            prompt = llm.calls[0]
            self.assertIn("Write your entire response in Chinese.", prompt)
            self.assertIn("决策摘要", prompt)
            self.assertIn("反馈快照", prompt)
            self.assertIn("关键约束", prompt)

    def test_risk_team_history_keeps_real_snapshot_blocks(self):
        llm = _CapturingLLM()
        node = create_aggressive_debator(llm)

        risk_debate_state = node(copy.deepcopy(self.base_state))["risk_debate_state"]
        formatted = format_risk_management_history(risk_debate_state)

        self.assertIn("决策摘要:", formatted)
        self.assertIn("- 评级: 持有", formatted)
        self.assertNotIn("##### 本轮复盘", formatted)
        self.assertNotIn("##### 自动复盘", formatted)
        self.assertIn("- 立场: x", formatted)
        self.assertIn("- 本轮新增与反驳: y；z；r", formatted)
        self.assertNotIn("决策摘要", risk_debate_state["current_aggressive_response"])
        self.assertNotIn("反馈快照", risk_debate_state["current_aggressive_response"])

    def test_portfolio_manager_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_portfolio_manager(llm, _EmptyMemory())
        node(self.base_state)

        prompt = llm.structured_calls[0]
        self.assertIn("Write your entire response in Chinese.", prompt)
        self.assertIn("反馈快照", prompt)
        self.assertIn("激进风险分析师", prompt)
        self.assertIn("保守风险分析师", prompt)
        self.assertIn("中性风险分析师", prompt)
        self.assertIn("评级体系", prompt)
        self.assertIn("买入", prompt)
        self.assertIn("增持", prompt)
        self.assertIn("持有", prompt)
        self.assertIn("减持", prompt)
        self.assertIn("卖出", prompt)
        self.assertIn("## 辩论结论", prompt)
        self.assertIn("## 行为逻辑", prompt)
        self.assertIn("## 持仓建议", prompt)
        self.assertIn("关键约束", prompt)
        self.assertNotIn("Lessons from past decisions", prompt)
        self.assertIn("催化节奏", prompt)
        self.assertNotIn("catalyst timing", prompt)

    def test_collaboration_stop_instruction_prefers_chinese_display(self):
        instruction = get_collaboration_stop_instruction()
        self.assertIn("最终交易建议: **买入/增持/持有/减持/卖出**", instruction)


if __name__ == "__main__":
    unittest.main()
