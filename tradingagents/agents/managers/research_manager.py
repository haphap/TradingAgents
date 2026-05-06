from tradingagents.agents.utils.agent_utils import (
    build_debate_brief,
    build_instrument_context,
    extract_feedback_snapshot,
    get_language_instruction,
    make_display_snapshot,
    get_snapshot_template,
    get_snapshot_writing_instruction,
    get_output_language,
    load_snapshot_file,
    localize_label,
    localize_rating_term,
    localize_role_name,
    normalize_chinese_manager_terms,
    save_snapshot_file,
    strip_feedback_snapshot,
    synthesize_side_report,
)
from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext


def _is_chinese_output() -> bool:
    return get_output_language().strip().lower() in {"chinese", "中文", "zh", "zh-cn", "zh-hans"}


def _research_action_logic_instruction() -> str:
    if _is_chinese_output():
        return (
            "- 说明估值、催化节奏、下行边界，以及确认 / 证伪信号如何共同导向你的决策。"
            " 每个触发条件必须给出具体数值或可验证标准，不能只写“等待确认”“观察变化”这类泛化表述。"
        )
    return "- Explain how valuation, catalyst timing, downside boundary, and confirmation / invalidation signals lead to your decision."


def _research_detail_instruction(section: str) -> str:
    if _is_chinese_output():
        if section == "conclusion":
            return "- 这一部分必须写成连贯分析段落，至少 4 句，不能只写简短观点或要点摘录。"
        return (
            "- 这一部分必须写成详细推理段落，至少 4 句，要把估值、催化节奏、价格信号和风险触发条件串成完整逻辑链。"
            " 对于持仓建议，必须给出以下具体标准：\n"
            "  (a) 关键支撑和阻力位的具体价格或均线位置（引用市场报告中的技术指标）；\n"
            "  (b) 成交量改善的具体阈值（如“成交量需达到近20日均量的1.3倍以上”）；\n"
            "  (c) 盈利验证的具体指标（如毛利率、订单增速、ROE的具体阈值）；\n"
            "  (d) 催化确认的具体条件（如“需看到季度订单增速环比提升5个百分点以上”）。"
        )
    if section == "conclusion":
        return "- Write this section as a coherent analysis paragraph with at least 4 full sentences; do not output terse fragments or simple bullet-style restatements."
    return (
        "- Write this section as a detailed reasoning paragraph with at least 4 full sentences, explicitly connecting valuation, catalysts, price action, and risk triggers to the recommendation."
        " For positioning recommendations, you MUST provide these specific criteria:\n"
        "  (a) Specific price levels or moving-average positions for key support/resistance (reference the market report);\n"
        "  (b) Specific volume-improvement thresholds (e.g., 'volume must reach 1.3x the 20-day average');\n"
        "  (c) Specific earnings-verification metrics (e.g., specific thresholds for gross margin, order growth, ROE);\n"
        "  (d) Specific catalyst-confirmation conditions (e.g., 'need to see quarterly order growth improve by 5pp QoQ')."
    )


def create_research_manager(llm, memory=None):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        research_report = state["research_report"]
        stock_report = state["stock_report"]

        investment_debate_state = state["investment_debate_state"]
        bull_snapshot_display = investment_debate_state.get("bull_snapshot", "")
        bear_snapshot_display = investment_debate_state.get("bear_snapshot", "")
        debate_brief = investment_debate_state.get("debate_brief", "")

        # Load full snapshots from files
        bull_snapshot_full = load_snapshot_file(investment_debate_state.get("bull_snapshot_path", "")) or bull_snapshot_display
        bear_snapshot_full = load_snapshot_file(investment_debate_state.get("bear_snapshot_path", "")) or bear_snapshot_display

        # Synthesize each side's full debate history into a comprehensive position report
        bull_history = investment_debate_state.get("bull_history", "")
        bear_history = investment_debate_state.get("bear_history", "")
        bull_report = synthesize_side_report(llm, "Bull Analyst", bull_history, bull_snapshot_full)
        bear_report = synthesize_side_report(llm, "Bear Analyst", bear_history, bear_snapshot_full)

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate the full multi-round debate and make a definitive decision: align with the {localize_role_name("Bear Analyst")}, the {localize_role_name("Bull Analyst")}, or choose {localize_rating_term("Hold")} only if it is strongly justified based on the arguments presented.

Your response must evaluate both sides before giving a position. Do not jump straight to the holding suggestion.
For ordinary lists, use Arabic numerals such as 1. 2. 3.; if you use Chinese section headings, keep forms like 一、二、三.

Use this exact output order with Markdown headings:
## {localize_label("Debate Conclusion", "辩论结论")}
- Assess which side presented the stronger case across the full debate, not just the latest exchange.
- Summarize the strongest points from both the {localize_role_name("Bull Analyst")} and the {localize_role_name("Bear Analyst")}.
- Explicitly point out the decisive weakness in the losing side's case.
{_research_detail_instruction("conclusion")}
- When writing in Chinese, use neutral investment wording such as "综合结论" and refer to the entire debate as "整场辩论"; avoid judicial wording like "判决" and avoid phrasing that sounds limited to "本轮".

## {localize_label("Action Logic", "行为逻辑")}
- Write your own decision logic from evidence to action, not just a repetition of either side.
{_research_action_logic_instruction()}
- This section must make clear what would cause you to maintain, add, reduce, or reverse the position.
{_research_detail_instruction("action")}

## {localize_label("Positioning Recommendation", "持仓建议")}
- Give a clear, actionable recommendation—{localize_rating_term("Buy")}, {localize_rating_term("Overweight")}, {localize_rating_term("Hold")}, {localize_rating_term("Underweight")}, or {localize_rating_term("Sell")}—grounded in the debate's strongest arguments.
- Include concrete execution guidance for the trader: entry / add / reduce conditions, risk controls, and what to monitor next.
- The rating field and the positioning recommendation text must point to the same action. Do not restate a different recommendation in prose.

Only after the three sections above, append a feedback block in this exact format. Do not place the feedback snapshot before the conclusion:
{get_snapshot_template()}
{get_snapshot_writing_instruction()}

{get_language_instruction()}
{instrument_context}

{localize_label("Rolling debate brief:", "滚动辩论摘要:")}
{debate_brief}

{localize_label("Bull Analyst comprehensive position report (synthesized from all rounds):", f"{localize_role_name('Bull Analyst')} 综合立场报告（基于全轮次辩论）:")}
{bull_report}

{localize_label("Bear Analyst comprehensive position report (synthesized from all rounds):", f"{localize_role_name('Bear Analyst')} 综合立场报告（基于全轮次辩论）:")}
{bear_report}

{localize_label("Industry research cross-analysis:", "行业研报交叉分析:")}
{research_report}

{localize_label("Individual stock research cross-analysis:", "个股研报交叉分析:")}
{stock_report}
"""
        normalized_content = normalize_chinese_manager_terms(
            invoke_structured_or_freetext(
                structured_llm,
                llm,
                prompt,
                render_research_plan,
                "Research Manager",
            )
        )
        judge_snapshot_full = extract_feedback_snapshot(normalized_content)
        debate_round = max(1, investment_debate_state.get("count", 0) // 2)
        judge_snapshot_path = save_snapshot_file(
            judge_snapshot_full,
            "Research Manager",
            state.get("company_of_interest", "unknown"),
            state.get("trade_date", "unknown"),
            debate_round,
        )
        judge_snapshot = make_display_snapshot(
            judge_snapshot_full, judge_snapshot_path
        )
        updated_brief = build_debate_brief(
            {
                "Bull Analyst": bull_snapshot_display,
                "Bear Analyst": bear_snapshot_display,
                "Research Manager": judge_snapshot,
            },
            latest_speaker="Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": normalized_content,
            "judge_snapshot": judge_snapshot,
            "judge_snapshot_path": judge_snapshot_path,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": strip_feedback_snapshot(normalized_content),
            "current_bull_response": investment_debate_state.get("current_bull_response", ""),
            "current_bear_response": investment_debate_state.get("current_bear_response", ""),
            "bull_snapshot": bull_snapshot_display,
            "bear_snapshot": bear_snapshot_display,
            "bull_snapshot_path": investment_debate_state.get("bull_snapshot_path", ""),
            "bear_snapshot_path": investment_debate_state.get("bear_snapshot_path", ""),
            "debate_brief": updated_brief,
            "latest_speaker": "Research Manager",
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": normalized_content,
        }

    return research_manager_node
