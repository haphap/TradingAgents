import unittest

from cli.main import MessageBuffer, format_research_team_history, format_risk_management_history
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
        self.assertIn("#### 多头分析师\n\n第一轮多头观点", formatted)
        self.assertIn("反馈快照:\n- 立场: 买入", formatted)
        self.assertIn("- 本轮新增与反驳: 强化多头", formatted)
        self.assertIn("金价走强", formatted)
        self.assertIn("估值担忧可控", formatted)
        self.assertIn("- 待验证: 跟踪量价", formatted)
        self.assertIn("#### 空头分析师\n\n第一轮空头观点", formatted)
        self.assertIn("### 第 2 轮", formatted)
        self.assertIn("#### 多头分析师\n\n第二轮多头补充", formatted)
        self.assertIn("更激进", formatted)
        self.assertIn("避险升级", formatted)
        self.assertIn("回撤是买点", formatted)
        self.assertIn("- 待验证: 盯并购兑现", formatted)
        self.assertIn("#### 空头分析师\n\n第二轮空头反驳", formatted)
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
                "DECISION SUMMARY:\n"
                "- Rating: SELL\n"
                "- Confidence: 70%\n"
                "- Time Horizon: 1-3 months\n"
                "- Key Assumptions:\n"
                "  1. Momentum remains weak.\n"
                "  2. Liquidity deteriorates.\n"
                "  3. No upside catalyst.\n"
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
        self.assertIn("#### 激进风险分析师\n\nRound 1 aggressive case", formatted)
        self.assertIn("决策摘要:\n- 评级: SELL", formatted)
        self.assertIn("FEEDBACK SNAPSHOT:\n- Stance: Sell", formatted)
        self.assertIn("- New this round & rebuttal: More defensive", formatted)
        self.assertIn("Momentum broke", formatted)
        self.assertIn("Upside is capped", formatted)
        self.assertIn("- To verify: Watch liquidity", formatted)
        self.assertIn("#### 保守风险分析师\n\nRound 1 conservative case", formatted)
        self.assertIn("#### 中性风险分析师\n\nRound 1 neutral case", formatted)
        self.assertIn("### 第 2 轮", formatted)
        self.assertIn("#### 激进风险分析师\n\nRound 2 aggressive follow-up", formatted)
        self.assertIn("### 投资组合经理结论\nPortfolio Manager: Final allocation", formatted)

    def test_inferred_snapshot_shows_snapshot_without_review_heading(self):
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

        self.assertIn("反馈快照:\n- 立场:", formatted)
        self.assertNotIn("##### 自动复盘", formatted)
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
        self.assertIn("## 辩论结论", formatted)
        self.assertIn("## 行为逻辑", formatted)
        self.assertIn("## 持仓建议", formatted)
        self.assertIn("#### 反馈快照摘要", formatted)
        self.assertIn("本轮新增与反驳", formatted)
        self.assertIn("需求验证节奏的约束", formatted)
        self.assertIn("空头高估了估值压缩速度", formatted)
        self.assertNotIn("(完整内容见:", formatted)
        self.assertLess(formatted.index("## 辩论结论"), formatted.index("#### 反馈快照摘要"))

    def test_research_manager_dedupes_repeated_feedback_snapshots(self):
        repeated_snapshot = (
            "反馈快照:\n"
            "- 立场: 增持——需求与盈利兑现仍占优。\n"
            "- 本轮新增与反驳: 新增了对需求验证节奏的约束。\n"
            "- 待验证: 跟踪订单、毛利率和客户资本开支。"
        )
        debate_state = {
            "bull_history": "",
            "bear_history": "",
            "judge_decision": (
                "## 辩论结论\n"
                "多头证据更扎实，空头对估值风险的论证不足。\n\n"
                "## 行为逻辑\n"
                "先验证需求兑现，再决定是否继续加仓。\n\n"
                "## 持仓建议\n"
                "维持增持，回踩支撑再分批加仓。\n\n"
                f"{repeated_snapshot}\n\n{repeated_snapshot}"
            ),
        }

        formatted = format_research_team_history(debate_state)

        self.assertEqual(formatted.count("反馈快照:"), 0)
        self.assertEqual(formatted.count("#### 反馈快照摘要"), 1)

    def test_research_manager_normalizes_judicial_wording_to_debate_conclusion(self):
        debate_state = {
            "bull_history": "",
            "bear_history": "",
            "judge_decision": (
                "## 辩论裁决\n"
                "判决结果：本轮双方论点势均力敌。\n\n"
                "## 行为逻辑\n"
                "等待更多盈利兑现信号后再提高仓位。"
            ),
        }

        formatted = format_research_team_history(debate_state)

        self.assertIn("## 辩论结论", formatted)
        self.assertIn("综合结论：整场辩论中双方论据势均力敌。", formatted)
        self.assertNotIn("判决结果", formatted)
        self.assertNotIn("本轮双方论点势均力敌", formatted)

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
        self.assertIn("决策摘要:\n- 评级: 增持", formatted)
        self.assertEqual(formatted.count("决策摘要:"), 1)
        self.assertNotIn("##### 决策摘要", formatted)
        self.assertLess(formatted.index("这是正文论证。"), formatted.index("决策摘要:\n- 评级: 增持"))

    def test_risk_history_strips_duplicate_role_self_introduction(self):
        risk_state = {
            "aggressive_history": "",
            "conservative_history": (
                "保守分析师: 作为保守风险分析师，我必须再次强调高估值和库存风险。\n"
                "反馈快照:\n"
                "- 当前观点: 持有\n"
                "- 发生了什么变化: 维持谨慎。\n"
                "- 为什么变化: 风险收益比偏弱。\n"
                "- 关键反驳: 不宜追高。\n"
                "- 下一轮教训: 跟踪支撑位。"
            ),
            "neutral_history": "",
            "judge_decision": "",
        }

        formatted = format_risk_management_history(risk_state)

        self.assertIn("#### 保守风险分析师\n\n我必须再次强调高估值和库存风险。", formatted)
        self.assertNotIn("#### 保守风险分析师\n\n作为保守风险分析师", formatted)

    def test_portfolio_manager_hides_snapshot_summary(self):
        risk_state = {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": (
                "Portfolio Manager: Final allocation\n\n"
                "FEEDBACK SNAPSHOT:\n"
                "- Stance: Hold\n"
                "- New this round & rebuttal: Keep flexibility.\n"
                "- To verify: Watch earnings."
            ),
            "judge_snapshot_path": "/tmp/portfolio_manager_round_1.md",
        }

        formatted = format_risk_management_history(risk_state)

        self.assertIn("### 投资组合经理结论\nPortfolio Manager: Final allocation", formatted)
        self.assertNotIn("反馈快照摘要", formatted)
        self.assertNotIn("FEEDBACK SNAPSHOT", formatted)

    def test_risk_management_history_can_hide_portfolio_manager_block(self):
        risk_state = {
            "aggressive_history": "Aggressive Analyst: Round 1 aggressive case",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": "Portfolio Manager: Final allocation",
        }

        formatted = format_risk_management_history(risk_state, include_manager=False)

        self.assertIn("#### 激进风险分析师", formatted)
        self.assertNotIn("### 投资组合经理结论", formatted)

    def test_portfolio_manager_normalizes_mixed_english_headings_and_terms(self):
        risk_state = {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "judge_decision": (
                "## Debate Verdict\n"
                "综合结论：应维持更谨慎的仓位。\n\n"
                "## Action Logic\n"
                "当前 time horizon 应控制在1-3个月，等待盈利确认后再决定是否加仓。\n\n"
                "## Positioning Recommendation\n"
                "维持持有，若支撑位失守则减仓。\n\n"
                "最终交易建议: **持有**"
            ),
        }

        formatted = format_risk_management_history(risk_state)

        self.assertIn("## 辩论结论", formatted)
        self.assertIn("## 行为逻辑", formatted)
        self.assertIn("## 持仓建议", formatted)
        self.assertIn("时间区间", formatted)
        self.assertNotIn("time horizon", formatted.lower())

    def test_message_buffer_localizes_fundamentals_section_title_in_chinese(self):
        buffer = MessageBuffer()
        buffer.init_for_analysis(["fundamentals"])
        buffer.update_report_section("fundamentals_report", "财务质量稳健。")

        self.assertIn("### 基本面分析\n财务质量稳健。", buffer.current_report)
        self.assertIn("### 基本面分析\n财务质量稳健。", buffer.final_report)
        self.assertNotIn("根本分析", buffer.final_report)


if __name__ == "__main__":
    unittest.main()
