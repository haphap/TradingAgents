import copy
import unittest

from cli.main import format_research_team_history, format_risk_management_history
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

    def invoke(self, prompt):
        self.calls.append(prompt)
        return _FakeResponse("测试输出\n反馈快照:\n- 当前观点: x\n- 发生了什么变化: y\n- 为什么变化: z\n- 关键反驳: r\n- 下一轮教训: l")


class _EmptyMemory:
    def get_memories(self, *_args, **_kwargs):
        return []


class OutputLanguagePropagationTests(unittest.TestCase):
    def setUp(self):
        self.original_config = copy.deepcopy(get_config())
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["output_language"] = "Chinese"
        set_config(cfg)

        self.base_state = {
            "company_of_interest": "002155.SZ",
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
                "count": 0,
            },
            "investment_debate_state": {
                "history": "",
                "bear_history": "",
                "bull_history": "",
                "current_response": "",
                "bull_snapshot": "",
                "bear_snapshot": "",
                "debate_brief": "",
                "latest_speaker": "",
                "count": 0,
            },
        }

    def tearDown(self):
        set_config(self.original_config)

    def test_trader_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_trader(llm, _EmptyMemory())
        node(self.base_state)

        system_prompt = llm.calls[0][0]["content"]
        self.assertIn("Write your entire response in Chinese.", system_prompt)

    def test_research_manager_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_research_manager(llm, _EmptyMemory())
        node(self.base_state)

        prompt = llm.calls[0]
        self.assertIn("Write your entire response in Chinese.", prompt)
        self.assertIn("多头分析师", prompt)
        self.assertIn("空头分析师", prompt)
        self.assertIn("买入", prompt)
        self.assertIn("持有", prompt)

    def test_bull_bear_researcher_prompts_require_chinese_body_and_decision_summary(self):
        for factory in (create_bull_researcher, create_bear_researcher):
            llm = _CapturingLLM()
            node = factory(llm, _EmptyMemory())
            node(self.base_state)

            prompt = llm.calls[0]
            self.assertIn("written entirely in Chinese", prompt)
            self.assertIn("决策摘要", prompt)
            self.assertIn("Internal lessons from similar situations", prompt)
            self.assertIn("do not quote, reveal, translate, or restate", prompt)
            self.assertIn("Write your entire response in Chinese.", prompt)
            self.assertIn("Do not use variants like", prompt)
            self.assertIn("牛派分析师", prompt)
            self.assertIn("熊派分析师", prompt)
            self.assertIn("反馈快照", prompt)
            self.assertIn("关键约束", prompt)

    def test_research_team_history_keeps_real_snapshot_blocks(self):
        llm = _CapturingLLM()
        node = create_bull_researcher(llm, _EmptyMemory())

        investment_debate_state = node(copy.deepcopy(self.base_state))["investment_debate_state"]
        formatted = format_research_team_history(investment_debate_state)

        self.assertIn("##### 本轮复盘", formatted)
        self.assertNotIn("##### 自动复盘", formatted)
        self.assertIn("- 立场: x", formatted)
        self.assertIn("- 本轮新增与反驳: y；z；r", formatted)
        self.assertNotIn("反馈快照", investment_debate_state["current_bull_response"])

    def test_normalize_chinese_role_terms_replaces_bull_bear_variants(self):
        text = "我是熊派分析师，也不同意牛派分析师和熊派投资者的说法。"
        normalized = normalize_chinese_role_terms(text)

        self.assertNotIn("熊派分析师", normalized)
        self.assertNotIn("牛派分析师", normalized)
        self.assertNotIn("熊派投资者", normalized)
        self.assertIn("空头分析师", normalized)
        self.assertIn("多头分析师", normalized)
        self.assertIn("空头投资者", normalized)

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
            self.assertIn("反馈快照", prompt)
            self.assertIn("关键约束", prompt)

    def test_risk_team_history_keeps_real_snapshot_blocks(self):
        llm = _CapturingLLM()
        node = create_aggressive_debator(llm)

        risk_debate_state = node(copy.deepcopy(self.base_state))["risk_debate_state"]
        formatted = format_risk_management_history(risk_debate_state)

        self.assertIn("##### 本轮复盘", formatted)
        self.assertNotIn("##### 自动复盘", formatted)
        self.assertIn("- 立场: x", formatted)
        self.assertIn("- 本轮新增与反驳: y；z；r", formatted)
        self.assertNotIn("反馈快照", risk_debate_state["current_aggressive_response"])

    def test_portfolio_manager_prompt_respects_output_language(self):
        llm = _CapturingLLM()
        node = create_portfolio_manager(llm, _EmptyMemory())
        node(self.base_state)

        prompt = llm.calls[0]
        self.assertIn("Write your entire response in Chinese.", prompt)
        self.assertIn("反馈快照", prompt)
        self.assertIn("激进分析师", prompt)
        self.assertIn("保守分析师", prompt)
        self.assertIn("中性分析师", prompt)
        self.assertIn("评级体系", prompt)
        self.assertIn("买入", prompt)
        self.assertIn("增持", prompt)
        self.assertIn("持有", prompt)
        self.assertIn("减持", prompt)
        self.assertIn("卖出", prompt)
        self.assertIn("关键约束", prompt)

    def test_collaboration_stop_instruction_prefers_chinese_display(self):
        instruction = get_collaboration_stop_instruction()
        self.assertIn("最终交易建议: **买入/增持/持有/减持/卖出**", instruction)


if __name__ == "__main__":
    unittest.main()
