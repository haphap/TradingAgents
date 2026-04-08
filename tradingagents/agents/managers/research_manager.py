from tradingagents.agents.utils.agent_utils import (
    build_debate_brief,
    build_instrument_context,
    extract_feedback_snapshot,
    get_language_instruction,
    get_snapshot_template,
    get_snapshot_writing_instruction,
    load_snapshot_file,
    localize_label,
    localize_rating_term,
    localize_role_name,
    normalize_chinese_role_terms,
    synthesize_side_report,
)


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

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

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision: align with the {localize_role_name("Bear Analyst")}, the {localize_role_name("Bull Analyst")}, or choose {localize_rating_term("Hold")} only if it is strongly justified based on the arguments presented.

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation—{localize_rating_term("Buy")}, {localize_rating_term("Sell")}, or {localize_rating_term("Hold")}—must be clear and actionable. Avoid defaulting to {localize_rating_term("Hold")} simply because both sides have valid points; commit to a stance grounded in the debate's strongest arguments.

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: A decisive stance supported by the most convincing arguments.
Rationale: An explanation of why these arguments lead to your conclusion.
Strategic Actions: Concrete steps for implementing the recommendation.
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Present your analysis conversationally, as if speaking naturally, without special formatting. 
After your analysis, append a feedback block in this exact format:
{get_snapshot_template()}
{get_snapshot_writing_instruction()}

Here are your past reflections on mistakes:
\"{past_memory_str}\"

{instrument_context}

{localize_label("Rolling debate brief:", "滚动辩论摘要:")}
{debate_brief}

{localize_label("Bull Analyst comprehensive position report (synthesized from all rounds):", f"{localize_role_name('Bull Analyst')} 综合立场报告（基于全轮次辩论）:")}
{bull_report}

{localize_label("Bear Analyst comprehensive position report (synthesized from all rounds):", f"{localize_role_name('Bear Analyst')} 综合立场报告（基于全轮次辩论）:")}
{bear_report}{get_language_instruction()}
"""
        response = llm.invoke(prompt)
        normalized_content = normalize_chinese_role_terms(response.content)
        judge_snapshot = extract_feedback_snapshot(normalized_content)
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
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": normalized_content,
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
