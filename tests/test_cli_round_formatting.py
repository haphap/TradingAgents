import unittest

from cli.main import format_research_team_history, format_risk_management_history
from tradingagents.dataflows.config import get_config, set_config
from tradingagents.default_config import DEFAULT_CONFIG


class CliRoundFormattingTests(unittest.TestCase):
    def setUp(self):
        self.original_config = get_config().copy()
        cfg = DEFAULT_CONFIG.copy()
        cfg["output_language"] = "Chinese"
        set_config(cfg)

    def tearDown(self):
        set_config(self.original_config)

    def test_research_team_history_is_grouped_by_round(self):
        debate_state = {
            "bull_history": (
                "多头分析师: 第一轮多头观点\n"
                "反馈快照:\n"
                "- 当前观点: 买入\n"
                "- 发生了什么变化: 强化多头\n"
                "- 为什么变化: 金价走强\n"
                "- 关键反驳: 估值担忧可控\n"
                "- 下一轮教训: 跟踪量价\n"
                "多头分析师: 第二轮多头补充\n"
                "反馈快照:\n"
                "- 当前观点: 强烈买入\n"
                "- 发生了什么变化: 更激进\n"
                "- 为什么变化: 避险升级\n"
                "- 关键反驳: 回撤是买点\n"
                "- 下一轮教训: 盯并购兑现"
            ),
            "bear_history": (
                "空头分析师: 第一轮空头观点\n"
                "反馈快照:\n"
                "- 当前观点: 持有\n"
                "- 发生了什么变化: 维持谨慎\n"
                "- 为什么变化: 估值偏高\n"
                "- 关键反驳: 上涨已透支\n"
                "- 下一轮教训: 看现金流\n"
                "空头分析师: 第二轮空头反驳\n"
                "反馈快照:\n"
                "- 当前观点: 减持\n"
                "- 发生了什么变化: 转向更谨慎\n"
                "- 为什么变化: 风险升高\n"
                "- 关键反驳: 高位放量\n"
                "- 下一轮教训: 盯库存"
            ),
            "judge_decision": "研究经理: 最终结论",
        }

        formatted = format_research_team_history(debate_state)

        self.assertIn("### 第 1 轮", formatted)
        self.assertIn("#### 多头分析师\n\n多头分析师: 第一轮多头观点", formatted)
        self.assertIn("##### 本轮复盘\n反馈快照:\n- 立场: 买入", formatted)
        self.assertIn("- 本轮新增与反驳: 强化多头", formatted)
        self.assertIn("金价走强", formatted)
        self.assertIn("估值担忧可控", formatted)
        self.assertIn("- 待验证: 跟踪量价", formatted)
        self.assertIn("#### 空头分析师\n\n空头分析师: 第一轮空头观点", formatted)
        self.assertIn("### 第 2 轮", formatted)
        self.assertIn("#### 多头分析师\n\n多头分析师: 第二轮多头补充", formatted)
        self.assertIn("更激进", formatted)
        self.assertIn("避险升级", formatted)
        self.assertIn("回撤是买点", formatted)
        self.assertIn("- 待验证: 盯并购兑现", formatted)
        self.assertIn("#### 空头分析师\n\n空头分析师: 第二轮空头反驳", formatted)
        self.assertIn("转向更谨慎", formatted)
        self.assertIn("风险升高", formatted)
        self.assertIn("高位放量", formatted)
        self.assertIn("- 待验证: 盯库存", formatted)
        self.assertIn("### 研究经理结论\n研究经理: 最终结论", formatted)
        self.assertIn("#### 反馈快照摘要", formatted)
        self.assertLess(formatted.index("研究经理: 最终结论"), formatted.index("#### 反馈快照摘要"))

    def test_risk_management_history_supports_english_prefixes(self):
        risk_state = {
            "aggressive_history": (
                "Aggressive Analyst: Round 1 aggressive case\n"
                "FEEDBACK SNAPSHOT:\n"
                "- Current thesis: Sell\n"
                "- What changed: More defensive\n"
                "- Why it changed: Momentum broke\n"
                "- Key rebuttal: Upside is capped\n"
                "- Lesson for next round: Watch liquidity\n"
                "Aggressive Analyst: Round 2 aggressive follow-up"
            ),
            "conservative_history": (
                "Conservative Analyst: Round 1 conservative case\n"
                "FEEDBACK SNAPSHOT:\n"
                "- Current thesis: Hold\n"
                "- What changed: Stayed cautious\n"
                "- Why it changed: Valuation rich\n"
                "- Key rebuttal: Do not chase\n"
                "- Lesson for next round: Check earnings"
            ),
            "neutral_history": (
                "Neutral Analyst: Round 1 neutral case\n"
                "FEEDBACK SNAPSHOT:\n"
                "- Current thesis: Hold\n"
                "- What changed: Balanced both sides\n"
                "- Why it changed: Conflicting signals\n"
                "- Key rebuttal: Need confirmation\n"
                "- Lesson for next round: Wait for breakout"
            ),
            "judge_decision": "Portfolio Manager: Final allocation",
        }

        formatted = format_risk_management_history(risk_state)

        self.assertIn("### 第 1 轮", formatted)
        self.assertIn("#### 激进分析师\n\nAggressive Analyst: Round 1 aggressive case", formatted)
        self.assertIn("##### 本轮复盘\nFEEDBACK SNAPSHOT:\n- Stance: Sell", formatted)
        self.assertIn("- New this round & rebuttal: More defensive", formatted)
        self.assertIn("Momentum broke", formatted)
        self.assertIn("Upside is capped", formatted)
        self.assertIn("- To verify: Watch liquidity", formatted)
        self.assertIn("#### 保守分析师\n\nConservative Analyst: Round 1 conservative case", formatted)
        self.assertIn("#### 中性分析师\n\nNeutral Analyst: Round 1 neutral case", formatted)
        self.assertIn("### 第 2 轮", formatted)
        self.assertIn("#### 激进分析师\n\nAggressive Analyst: Round 2 aggressive follow-up", formatted)
        self.assertIn("### 投资组合经理结论\nPortfolio Manager: Final allocation", formatted)

    def test_inferred_snapshot_uses_auto_review_title(self):
        debate_state = {
            "bull_history": (
                "多头分析师: 本轮新增了对库存风险的反驳，并强调需要继续跟踪金价与并购进度。\n"
                "反馈快照:\n"
                "- 当前观点: 暂无。\n"
                "- 发生了什么变化: 未明确说明。\n"
                "- 为什么变化: 未明确说明。\n"
                "- 关键反驳: 未明确说明。\n"
                "- 下一轮教训: 未明确说明。"
            ),
            "bear_history": "",
            "judge_decision": "",
        }

        formatted = format_research_team_history(debate_state)

        self.assertIn("##### 自动复盘", formatted)
        self.assertNotIn("##### 本轮复盘", formatted)

    def test_research_manager_shows_body_before_snapshot_summary(self):
        debate_state = {
            "bull_history": "",
            "bear_history": "",
            "judge_decision": (
                "## 辩论裁决\n"
                "多头证据更扎实，空头对估值风险的论证不足。\n\n"
                "## 行为逻辑\n"
                "先验证需求兑现，再决定是否继续加仓。\n\n"
                "## 持仓建议\n"
                "维持增持，回踩支撑再分批加仓。\n\n"
                "反馈快照:\n"
                "- 立场: 增持——需求与盈利兑现仍占优。\n"
                "- 本轮新增: 新增了对需求验证节奏的约束。\n"
                "- 关键反驳: 空头高估了估值压缩速度。\n"
                "- 待验证: 跟踪订单、毛利率和客户资本开支。"
            ),
            "judge_snapshot_path": "/tmp/research_manager_round_1.md",
        }

        formatted = format_research_team_history(debate_state)

        self.assertIn("### 研究经理结论", formatted)
        self.assertIn("## 辩论裁决", formatted)
        self.assertIn("## 行为逻辑", formatted)
        self.assertIn("## 持仓建议", formatted)
        self.assertIn("#### 反馈快照摘要", formatted)
        self.assertIn("本轮新增与反驳", formatted)
        self.assertIn("需求验证节奏的约束", formatted)
        self.assertIn("空头高估了估值压缩速度", formatted)
        self.assertIn("(完整内容见: /tmp/research_manager_round_1.md)", formatted)
        self.assertLess(formatted.index("## 辩论裁决"), formatted.index("#### 反馈快照摘要"))

    def test_research_team_history_shows_decision_summary_outside_argument_body(self):
        debate_state = {
            "bull_history": (
                "多头分析师: 这是正文论证。\n\n"
                "决策摘要:\n"
                "- 评级: 增持\n"
                "- 置信度: 80%\n"
                "- 时间区间: 6-12个月\n"
                "- 关键假设:\n"
                "  1. AI需求持续。\n"
                "  2. 公司维持份额。\n"
                "  3. 供应链稳定。\n\n"
                "反馈快照:\n"
                "- 当前观点: 增持\n"
                "- 发生了什么变化: 新增催化剂。\n"
                "- 为什么变化: 需求增强。\n"
                "- 关键反驳: 空头低估景气度。\n"
                "- 下一轮教训: 跟踪订单。"
            ),
            "bear_history": "",
            "judge_decision": "",
        }

        formatted = format_research_team_history(debate_state)

        self.assertIn("这是正文论证。", formatted)
        self.assertIn("##### 决策摘要\n决策摘要:\n- 评级: 增持", formatted)
        self.assertLess(formatted.index("这是正文论证。"), formatted.index("##### 决策摘要"))


if __name__ == "__main__":
    unittest.main()
