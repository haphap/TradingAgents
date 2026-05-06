from tradingagents.agents.utils.agent_utils import (
    build_debate_brief,
    build_instrument_context,
    extract_feedback_snapshot,
    get_language_instruction,
    get_localized_final_proposal_instruction,
    get_localized_rating_scale,
    get_snapshot_template,
    get_snapshot_writing_instruction,
    get_output_language,
    load_snapshot_file,
    localize_label,
    localize_rating_term,
    localize_role_name,
    make_display_snapshot,
    normalize_chinese_manager_terms,
    save_snapshot_file,
    synthesize_side_report,
)
from tradingagents.agents.schemas import PortfolioDecision, render_portfolio_decision
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext


def _is_chinese_output() -> bool:
    return get_output_language().strip().lower() in {"chinese", "中文", "zh", "zh-cn", "zh-hans"}


def _portfolio_action_logic_instruction() -> str:
    if _is_chinese_output():
        return (
            "- 说明估值、催化节奏、下行边界、仓位大小以及加仓 / 减仓 / 对冲触发条件如何共同导向你的决策。"
            ' 每个触发条件必须给出具体数值或可验证标准，不能只写\u201c等待确认\u201d\u201c观察成交量\u201d这类泛化表述。'
        )
    return (
        "- Explain how valuation, catalyst timing, downside boundary, position sizing, and add / reduce / hedge triggers lead to your decision."
        " Every trigger condition must cite a specific number or verifiable criterion — do not use vague phrases like 'wait for confirmation' or 'watch volume'."
    )


def _portfolio_detail_instruction(section: str) -> str:
    if _is_chinese_output():
        if section == "conclusion":
            return "- 这一部分必须写成连贯分析段落，至少 4 句，不能只写简短观点或要点摘录。"
        return (
            "- 这一部分必须写成详细推理段落，至少 4 句，要把估值、催化节奏、仓位大小以及对冲 / 减仓触发条件串成完整逻辑链。"
            " 对于执行计划和风险控制，必须给出以下具体标准：\n"
            "  (a) 关键支撑和阻力位的具体价格或均线位置（引用市场报告中的技术指标，如50日均线、布林中轨、前低或密集成交区）；\n"
            '  (b) 成交量改善的具体阈值（相对近5日或20日均量达到什么倍数，如\u201c成交量需达到近20日均量的1.3倍以上\u201d）；\n'
            "  (c) 盈利验证的具体指标（如毛利率需维持在多少以上、订单增速需达到多少、ROE需高于多少）；\n"
            '  (d) 催化确认的具体条件（如\u201c需看到季度订单增速环比提升5个百分点以上\u201d或\u201c需看到行业政策落地\u201d）。'
        )
    if section == "conclusion":
        return "- Write this section as a coherent analysis paragraph with at least 4 full sentences; do not output terse fragments or simple bullet-style restatements."
    return (
        "- Write this section as a detailed reasoning paragraph with at least 4 full sentences, explicitly connecting valuation, catalyst timing, position sizing, and hedge / reduce triggers to the recommendation."
        " For execution plan and risk controls, you MUST provide these specific criteria:\n"
        "  (a) Specific price levels or moving-average positions for key support/resistance (reference the market report — e.g., 50-day SMA, Bollinger mid-band, prior swing low, dense volume area);\n"
        "  (b) Specific volume-improvement thresholds (e.g., 'volume must reach 1.3x the 20-day average');\n"
        "  (c) Specific earnings-verification metrics (e.g., 'gross margin must stay above X%', 'order growth must exceed Y%', 'ROE must be above Z%');\n"
        "  (d) Specific catalyst-confirmation conditions (e.g., 'need to see quarterly order growth improve by 5pp QoQ' or 'need to see industry policy enacted')."
    )


def create_portfolio_manager(llm, memory=None):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:

        instrument_context = build_instrument_context(state["company_of_interest"])
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        research_report = state["research_report"]
        stock_report = state["stock_report"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]
        aggressive_snapshot_display = risk_debate_state.get("aggressive_snapshot", "")
        conservative_snapshot_display = risk_debate_state.get("conservative_snapshot", "")
        neutral_snapshot_display = risk_debate_state.get("neutral_snapshot", "")
        debate_brief = risk_debate_state.get("debate_brief", "")

        # Load full snapshots from files
        aggressive_snapshot_full = load_snapshot_file(risk_debate_state.get("aggressive_snapshot_path", "")) or aggressive_snapshot_display
        conservative_snapshot_full = load_snapshot_file(risk_debate_state.get("conservative_snapshot_path", "")) or conservative_snapshot_display
        neutral_snapshot_full = load_snapshot_file(risk_debate_state.get("neutral_snapshot_path", "")) or neutral_snapshot_display

        # Synthesize each analyst's full debate history into a comprehensive position report
        aggressive_report = synthesize_side_report(llm, "Aggressive Analyst", risk_debate_state.get("aggressive_history", ""), aggressive_snapshot_full)
        conservative_report = synthesize_side_report(llm, "Conservative Analyst", risk_debate_state.get("conservative_history", ""), conservative_snapshot_full)
        neutral_report = synthesize_side_report(llm, "Neutral Analyst", risk_debate_state.get("neutral_history", ""), neutral_snapshot_full)
        past_context = state.get("past_context", "").strip()
        lessons_block = ""
        if past_context:
            lessons_header = (
                "历史决策复盘（仅供内部吸收，不要照抄到可见答案中）"
                if _is_chinese_output()
                else "Lessons from resolved past decisions (internal only; do not quote verbatim)"
            )
            lessons_block = f"**{lessons_header}:**\n{past_context}\n\n"

        prompt = f"""As the Portfolio Manager, synthesize the full risk debate and deliver the final trading decision.

Your response must evaluate all three risk perspectives before giving a position. Do not jump straight to the final recommendation.
For ordinary lists, use Arabic numerals such as 1. 2. 3.; if you use Chinese section headings, keep forms like 一、二、三.

Use this exact output order with Markdown headings:
## {localize_label("Debate Conclusion", "辩论结论")}
- Assess which risk perspective presented the strongest case across the full debate.
- Summarize the strongest points from the {localize_role_name("Aggressive Analyst")}, {localize_role_name("Conservative Analyst")}, and {localize_role_name("Neutral Analyst")}.
- Explain the decisive weakness in the view you did not ultimately follow, or clarify why multiple views were overruled.
{_portfolio_detail_instruction("conclusion")}

## {localize_label("Action Logic", "行为逻辑")}
- Write your own decision logic from evidence to execution, not just a paraphrase of one analyst.
{_portfolio_action_logic_instruction()}
- Make clear what would cause you to maintain, add, reduce, hedge, or reverse the position.
{_portfolio_detail_instruction("action")}

## {localize_label("Positioning Recommendation", "持仓建议")}
- Give a clear, actionable recommendation—{localize_rating_term("Buy")}, {localize_rating_term("Overweight")}, {localize_rating_term("Hold")}, {localize_rating_term("Underweight")}, or {localize_rating_term("Sell")}—grounded in the debate's strongest evidence.
- Include concrete execution guidance: entry / add / reduce conditions, maximum initial sizing, risk controls, and what to monitor next.
- When writing in Chinese, avoid mixed English labels such as "Time Horizon", "Executive Summary", or "Investment Thesis".
- The rating, the positioning recommendation text, and the final transaction proposal must all point to the same action. Do not restate a conflicting recommendation in prose.

{instrument_context}

---

{get_localized_rating_scale()}

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_block.strip()}

**{localize_label("Rolling Risk Debate Brief", "滚动风险辩论摘要")}:**
{debate_brief}

**{localize_label("Aggressive Analyst comprehensive position report (synthesized from all rounds)", f"{localize_role_name('Aggressive Analyst')} 综合立场报告（基于全轮次辩论）")}:**
{aggressive_report}

**{localize_label("Conservative Analyst comprehensive position report (synthesized from all rounds)", f"{localize_role_name('Conservative Analyst')} 综合立场报告（基于全轮次辩论）")}:**
{conservative_report}

**{localize_label("Neutral Analyst comprehensive position report (synthesized from all rounds)", f"{localize_role_name('Neutral Analyst')} 综合立场报告（基于全轮次辩论）")}:**
{neutral_report}

{localize_label("Industry research cross-analysis:", "行业研报交叉分析:")}
{research_report}

{localize_label("Individual stock research cross-analysis:", "个股研报交叉分析:")}
{stock_report}

Be decisive and ground every conclusion in specific evidence from the analysts. {get_localized_final_proposal_instruction()}
Only after the three sections above and the final transaction proposal line, append a feedback block in this exact format:
{get_snapshot_template()}
{get_snapshot_writing_instruction()}{get_language_instruction()}"""

        normalized_content = normalize_chinese_manager_terms(
            invoke_structured_or_freetext(
                structured_llm,
                llm,
                prompt,
                render_portfolio_decision,
                "Portfolio Manager",
            )
        )
        judge_snapshot_full = extract_feedback_snapshot(normalized_content)
        debate_round = max(1, (risk_debate_state.get("count", 0) + 2) // 3)
        judge_snapshot_path = save_snapshot_file(
            judge_snapshot_full,
            "Portfolio Manager",
            state.get("company_of_interest", "unknown"),
            state.get("trade_date", "unknown"),
            debate_round,
        )
        judge_snapshot = make_display_snapshot(judge_snapshot_full, judge_snapshot_path)
        updated_brief = build_debate_brief(
            {
                "Aggressive Analyst": aggressive_snapshot_display,
                "Conservative Analyst": conservative_snapshot_display,
                "Neutral Analyst": neutral_snapshot_display,
                "Portfolio Manager": judge_snapshot,
            },
            latest_speaker="Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": normalized_content,
            "history": risk_debate_state.get("history", ""),
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "debate_brief": updated_brief,
            "latest_speaker": "Portfolio Manager",
            "current_aggressive_response": risk_debate_state.get("current_aggressive_response", ""),
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get("current_neutral_response", ""),
            "aggressive_snapshot": aggressive_snapshot_display,
            "conservative_snapshot": conservative_snapshot_display,
            "neutral_snapshot": neutral_snapshot_display,
            "aggressive_snapshot_path": risk_debate_state.get("aggressive_snapshot_path", ""),
            "conservative_snapshot_path": risk_debate_state.get("conservative_snapshot_path", ""),
            "neutral_snapshot_path": risk_debate_state.get("neutral_snapshot_path", ""),
            "judge_snapshot": judge_snapshot,
            "judge_snapshot_path": judge_snapshot_path,
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": normalized_content,
        }

    return portfolio_manager_node
