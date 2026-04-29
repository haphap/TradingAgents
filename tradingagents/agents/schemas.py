import re
from enum import Enum

from pydantic import BaseModel, Field

from tradingagents.agents.utils.agent_utils import (
    get_output_language,
    localize_label,
    localize_rating_term,
)
from tradingagents.agents.utils.rating import detect_chinese_rating, detect_english_rating


class PortfolioRating(str, Enum):
    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class ResearchPlan(BaseModel):
    debate_conclusion: str = Field(description="A detailed synthesis paragraph comparing both sides, naming the strongest evidence from each, and explaining the decisive weakness in the losing view.")
    action_logic: str = Field(description="A detailed evidence-to-action paragraph linking valuation, catalysts, downside boundaries, and confirmation/invalidation triggers to the final decision.")
    positioning_recommendation: str = Field(description="Actionable trading guidance with execution details, sizing logic, concrete add/reduce conditions, and monitoring priorities.")
    rating: PortfolioRating = Field(description="Final research-manager rating.")
    snapshot_stance: str = Field(description="Concise stance for the feedback snapshot.")
    snapshot_new_and_rebuttal: str = Field(description="What was newly added this round and how it rebuts the opposing case.")
    snapshot_to_verify: str = Field(description="Specific follow-up points or triggers to verify next.")


class TraderProposal(BaseModel):
    thesis: str = Field(description="Concise trading thesis explaining the proposed action.")
    execution_plan: str = Field(description="Concrete execution plan with support/resistance references, volume thresholds, catalyst triggers, and explicit add/reduce/exit conditions.")
    risk_management: str = Field(description="Risk controls, invalidation signals, monitoring thresholds, and the actions to take when those thresholds are breached.")
    rating: PortfolioRating = Field(description="Trader recommendation.")


class PortfolioDecision(BaseModel):
    debate_conclusion: str = Field(description="A detailed synthesis of the full risk debate across all perspectives, explicitly comparing aggressive, conservative, and neutral cases and stating why the losing view was overruled.")
    action_logic: str = Field(description="A detailed portfolio-manager paragraph showing how evidence leads to sizing, hedging, add/reduce triggers, and risk controls.")
    positioning_recommendation: str = Field(description="Final actionable portfolio recommendation and implementation guidance with sizing, execution sequence, and monitoring priorities.")
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


def _expected_rating_text(rating: PortfolioRating) -> str:
    return localize_rating_term(rating.value) if _is_chinese_output() else rating.value.upper()


def _compact_text(text: str) -> str:
    return re.sub(r"[\s:：,，。.!！？/\-—_()（）]+", "", text or "").strip().lower()


def _is_placeholder_like(text: str) -> bool:
    compact = _compact_text(text)
    if not compact:
        return True

    exact_placeholders = {
        "评估双方论证强度总结核心论点与致命弱点",
        "估值催化节奏下行边界与确认证伪信号的推演路径",
        "明确评级与执行指引",
        "本轮新增与反驳",
        "待验证",
        "balancedconclusionafterevaluatingbothbullandbearcases",
        "evidencetoactionlogicexplainingvaluationcatalystsrisksandtriggers",
        "actionabletradingguidancewithexecutiondetails",
        "synthesisofthefullriskdebateacrossallperspectives",
        "portfoliomanagerlogicfromevidencetosizingandexecution",
        "finalactionableportfoliorecommendationandimplementationguidance",
        "concisestanceforthefeedbacksnapshot",
        "whatwasnewlyaddedthisroundandhowitrebutstheopposingcase",
        "specificfollowuppointsortriggerstoverifynext",
        "whatwasnewlyaddedthisroundandhowitrebuttedcompetingviews",
        "specificitemsortriggerstoverifynext",
        "concisetradingthesisexplainingtheproposedaction",
        "concreteexecutionplanwithentryaddreduceorexitconditions",
        "riskcontrolsinvalidationsignalsandmonitoringitems",
    }
    if compact in exact_placeholders:
        return True

    placeholder_patterns = (
        r"评估.*论证强度.*总结.*核心论点.*致命弱点",
        r"估值.*催化节奏.*下行边界.*确认.*证伪信号",
        r"明确评级.*执行指引",
        r"balanced.*bull.*bear.*cases",
        r"evidence.*action.*logic.*valuation.*catalysts",
        r"actionable.*guidance.*execution.*details",
        r"synthesis.*risk.*debate",
        r"portfolio.*logic.*evidence.*execution",
        r"concrete.*execution.*entry.*reduce.*exit",
        r"risk.*controls.*monitoring.*items",
    )
    return any(re.search(pattern, compact, re.IGNORECASE) for pattern in placeholder_patterns)


def _contains_explicit_rating_marker(text: str) -> bool:
    if not text:
        return False
    if _is_chinese_output():
        return any(
            marker in text
            for marker in ("建议评级", "评级", "最终交易建议", "建议买入", "建议增持", "建议持有", "建议减持", "建议卖出", "维持买入", "维持增持", "维持持有", "维持减持", "维持卖出", "采取买入策略", "采取增持策略", "采取持有策略", "采取减持策略", "采取卖出策略")
        )
    upper_text = text.upper()
    return any(
        marker in upper_text
        for marker in ("RECOMMENDATION:", "RATING:", "FINAL TRANSACTION PROPOSAL:", "RECOMMEND BUY", "RECOMMEND OVERWEIGHT", "RECOMMEND HOLD", "RECOMMEND UNDERWEIGHT", "RECOMMEND SELL", "MAINTAIN BUY", "MAINTAIN OVERWEIGHT", "MAINTAIN HOLD", "MAINTAIN UNDERWEIGHT", "MAINTAIN SELL")
    )


def _contains_any_rating_term(text: str) -> bool:
    if not text:
        return False
    if _is_chinese_output():
        return any(term in text for term in ("买入", "增持", "持有", "减持", "卖出"))
    upper_text = text.upper()
    return any(term in upper_text for term in ("BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"))


def _is_recommendation_only_segment(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False
    if _is_chinese_output():
        patterns = (
            r"^(?:建议评级|评级|最终交易建议)[:：].+$",
            r"^(?:建议评级|评级|最终交易建议)(?:为)?(?:买入|增持|持有|减持|卖出)[。！!]*$",
            r"^针对[^，。,；;]+[，,]?(?:建议|应|宜)(?:采取)?(?:买入|增持|持有|减持|卖出)(?:策略)?[。！!]*$",
            r"^(?:建议|维持|转为)(?:买入|增持|持有|减持|卖出)(?:策略)?[。！!]*$",
            r"^建议采取(?:买入|增持|持有|减持|卖出)策略[。！!]*$",
        )
    else:
        patterns = (
            r"^(?:RECOMMENDATION|RATING|FINALTRANSACTIONPROPOSAL)[:：].+$",
            r"^(?:RECOMMEND|MAINTAIN|SHIFTTO|MOVETO)(?:BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)[.!]*$",
            r"^FOR[A-Z0-9._-]+,?(?:RECOMMEND|MAINTAIN)(?:BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)[.!]*$",
        )
    return any(re.match(pattern, compact, re.IGNORECASE) for pattern in patterns)


def _has_conditional_prefix(prefix: str) -> bool:
    return bool(re.search(r"(若|如果|如|待|当|一旦|if|when|unless)", prefix, re.IGNORECASE))


def _has_conflicting_primary_action(text: str, rating: PortfolioRating) -> bool:
    content = (text or "").strip()
    if not content:
        return False

    if _is_chinese_output():
        bullish_pattern = re.compile(r"(买入|建仓|加仓|增持|提高仓位|扩大仓位|回补)")
        bearish_pattern = re.compile(r"(减持|减仓|降低敞口|卖出|退出|清仓|止盈)")
    else:
        bullish_pattern = re.compile(r"(buy|build|add|increase exposure|top up|rebuild)", re.IGNORECASE)
        bearish_pattern = re.compile(r"(reduce|trim|sell|exit|cut exposure|take profit)", re.IGNORECASE)

    if rating in {PortfolioRating.BUY, PortfolioRating.OVERWEIGHT}:
        conflicting_pattern = bearish_pattern
    elif rating in {PortfolioRating.UNDERWEIGHT, PortfolioRating.SELL}:
        conflicting_pattern = bullish_pattern
    else:
        conflicting_pattern = re.compile(
            f"{bullish_pattern.pattern}|{bearish_pattern.pattern}",
            bullish_pattern.flags | bearish_pattern.flags,
        )

    for match in conflicting_pattern.finditer(content):
        prefix = content[max(0, match.start() - 24):match.start()]
        if _has_conditional_prefix(prefix):
            continue
        return True
    return False


def _sanitize_positioning_recommendation(text: str, rating: PortfolioRating) -> str:
    content = (text or "").strip()
    if not content:
        return _default_positioning_guidance(rating)

    lines = []
    for raw_line in re.split(r"\n+", content):
        line = raw_line.strip()
        if not line:
            continue
        if _contains_explicit_rating_marker(line) and _is_recommendation_only_segment(line):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    if _is_chinese_output():
        cleaned = re.sub(
            r"(?:^|\n)\s*针对[^\n。！？!?]{0,60}?(?:建议|应|宜)(?:采取)?(?:买入|增持|持有|减持|卖出)(?:策略)?[。！!?\n]*",
            "\n",
            cleaned,
        )
        cleaned = re.sub(
            r"(?:^|\n)\s*(?:建议|维持|转为)(?:买入|增持|持有|减持|卖出)(?:策略)?[。！!?\n]*",
            "\n",
            cleaned,
        )
        cleaned = re.sub(
            r"(?:^|\n)\s*(?:建议评级|评级|最终交易建议)(?:为)?\s*(?:买入|增持|持有|减持|卖出)[。！!?\n]*",
            "\n",
            cleaned,
        )
    else:
        cleaned = re.sub(
            r"(?:^|\n)\s*for [^\n.]{0,60}?(?:recommend|maintain|shift to|move to) (?:buy|overweight|hold|underweight|sell)[.!?\n]*",
            "\n",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"(?:^|\n)\s*(?:recommend|maintain|shift to|move to) (?:buy|overweight|hold|underweight|sell)[.!?\n]*",
            "\n",
            cleaned,
            flags=re.IGNORECASE,
        )

    segments = []
    for segment in re.split(r"(?<=[。！？!?\.])\s*", cleaned):
        sentence = segment.strip()
        if not sentence:
            continue
        if _contains_explicit_rating_marker(sentence) and _is_recommendation_only_segment(sentence):
            continue
        segments.append(sentence)
    cleaned = "\n".join(segments).strip()
    if _is_placeholder_like(cleaned) or _has_conflicting_primary_action(cleaned, rating):
        return _default_positioning_guidance(rating)
    return cleaned or _default_positioning_guidance(rating)


def _default_positioning_guidance(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "以分批建仓为主，并持续跟踪验证信号、仓位节奏与风险边界。",
            PortfolioRating.OVERWEIGHT: "在保留现有仓位基础上择机加仓，并持续跟踪验证信号、仓位节奏与风险边界。",
            PortfolioRating.HOLD: "维持当前仓位，等待新增验证信号后再决定是否加仓或减仓。",
            PortfolioRating.UNDERWEIGHT: "以降低敞口为主，分批减仓，并持续跟踪反弹强度与风险释放节奏。",
            PortfolioRating.SELL: "以退出仓位或避免入场为主，并持续跟踪风险是否重新定价。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "Build the position in stages while monitoring validation signals, sizing pace, and risk boundaries.",
        PortfolioRating.OVERWEIGHT: "Add selectively from an existing position while monitoring validation signals, sizing pace, and risk boundaries.",
        PortfolioRating.HOLD: "Maintain the current position and wait for stronger confirmation before changing exposure.",
        PortfolioRating.UNDERWEIGHT: "Reduce exposure in stages while monitoring rebound strength and the pace of risk release.",
        PortfolioRating.SELL: "Prioritize exiting or avoiding entry while monitoring whether risks are being repriced.",
    }
    return mapping[rating]


def _default_debate_conclusion(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "整场辩论中，看多一侧不仅更充分地证明了产业趋势、盈利兑现与价格承接之间的正向联动，也更清楚地解释了为何短期波动不足以破坏中期上行结构。相较之下，看空一侧虽然提示了估值与节奏风险，但未能证明这些风险已经足以推翻主线逻辑，因此当前结论更偏向积极布局而不是继续观望。",
            PortfolioRating.OVERWEIGHT: "整场辩论中，看多一侧在产业趋势、催化兑现与盈利韧性上的论证更占优，说明上行逻辑仍是主导变量；但看空一侧关于波动、估值与兑现节奏的提醒也提示仓位不宜一次性放大。综合来看，更合理的结论是在保留风险边界的前提下逐步增配，而不是激进满仓。",
            PortfolioRating.HOLD: "整场辩论中，多空双方都给出了成立的证据：乐观一侧证明了中期逻辑尚未被破坏，谨慎一侧则指出短期估值、节奏与价格确认仍不够充分。由于现阶段还缺少能够打破平衡的新证据，最稳妥的结论不是贸然加仓或减仓，而是先维持现有敞口并等待更明确的验证信号。",
            PortfolioRating.UNDERWEIGHT: "整场辩论中，偏谨慎一侧对估值约束、风险释放节奏和下行边界的论证更完整，说明当前风险重定价的压力尚未结束。即便乐观一侧提出了中长期逻辑，其关键前提仍依赖后续催化兑现与价格结构修复，因此当前更适合先降低敞口、把仓位收回到更安全的水平。",
            PortfolioRating.SELL: "整场辩论中，看空一侧对基本面下修、技术破位和风险收益失衡的论证最具决定性，并且更清楚地说明了继续持有的代价正在上升。相比之下，乐观论点仍停留在潜在修复或远期改善的假设上，尚不足以抵消当前下行风险，因此更合理的结论是退出仓位而不是继续承受回撤。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "Across the full debate, the bullish side made the more complete case on trend durability, earnings follow-through, and price support, and it also explained more convincingly why the recent risks are not yet thesis-breaking. The opposing side raised valid caution flags, but it did not show that those risks are strong enough to overturn the broader upside setup, so active accumulation is more justified than continued hesitation.",
        PortfolioRating.OVERWEIGHT: "Across the full debate, the bullish evidence is stronger on balance because the upside thesis still has better support from catalysts, earnings resilience, and market structure. Even so, the cautious side made a credible case that volatility and timing risk still matter, so the right conclusion is controlled upside exposure rather than an all-in posture.",
        PortfolioRating.HOLD: "Across the full debate, both sides surfaced credible evidence: the bullish camp showed that the core thesis is still intact, while the cautious camp showed that timing, valuation, and confirmation risk remain unresolved. Because neither side produced the decisive incremental evidence needed to justify a bigger exposure change, maintaining current positioning is more disciplined than forcing either an add or a reduction.",
        PortfolioRating.UNDERWEIGHT: "Across the full debate, the bearish side made the stronger case on valuation pressure, risk-release cadence, and downside boundaries, which means the market is still repricing risk rather than rewarding conviction. The bullish case still depends on future confirmation rather than present proof, so trimming exposure is more appropriate than defending a full-sized position.",
        PortfolioRating.SELL: "Across the full debate, the bearish side presented the decisive case on deteriorating fundamentals, technical breakdown risk, and an unfavorable risk-reward profile. The more optimistic view still relies on future stabilization rather than current evidence, so exiting is more appropriate than continuing to absorb drawdown while waiting for a thesis repair that has not yet materialized.",
    }
    return mapping[rating]


def _default_action_logic(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "当前决策建立在三个前提之上：估值仍处在可接受区间、核心催化处于持续兑现阶段、价格结构尚未出现趋势性破坏。因此执行上应以分批建仓为主，而不是一次性打满仓位；只有在成交量、盈利验证和催化进度继续共振改善时，才逐步放大敞口。若后续出现催化落空、量价背离或关键支撑失守，则应立刻放慢建仓节奏并重新审视原先的乐观假设。",
            PortfolioRating.OVERWEIGHT: "当前更适合增配而不是激进做多，因为上行主线仍在，但兑现节奏和波动约束要求仓位管理保持克制。执行上应围绕已有底仓分批增加敞口，把加仓动作与催化确认、估值消化和价格承接绑定在一起，而不是仅凭情绪推动扩大风险暴露。若后续出现业绩兑现不及预期、催化推迟或价格结构转弱，就应暂停增配并把仓位退回到更中性的水平。",
            PortfolioRating.HOLD: "当前更合理的动作是维持现有仓位，因为新增上行催化尚不足以支持继续加仓，而风险释放也未恶化到必须立刻减仓的程度。执行逻辑上，应把注意力放在关键支撑、成交量变化、估值消化与后续业绩验证上，等待这些变量给出更清晰的方向指引。若价格和基本面同步改善，可再讨论增加敞口；若支撑失守或盈利预期显著下修，则应转入更保守的减仓方案。",
            PortfolioRating.UNDERWEIGHT: "当前决策的核心不是彻底看空，而是在风险尚未充分释放之前主动降低净暴露，把组合重新拉回到更能承受波动的位置。执行上应优先减少弹性较大、验证最弱的仓位，同时保留对后续修复仍有把握的核心观察仓位，以便在条件改善时重新评估。若估值回到更合理区间、催化恢复兑现且价格结构止跌企稳，才考虑逐步回补，而不是提前逆势放大风险。",
            PortfolioRating.SELL: "当前风险收益比已经明显失衡，继续持有并不能换来足够的上行补偿，因此执行上应以退出仓位或避免入场为主。只有当基本面出现可验证修复、价格结构完成重建、并且市场重新给出稳定承接信号时，才值得重新建立跟踪仓位。在那之前，任何抄底动作都更像是在用仓位承担不必要的不确定性。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "The current decision works only if three conditions continue to hold together: valuation remains supportive, catalysts keep improving, and price structure does not break down. That means execution should favor staged accumulation rather than a one-shot position, with adds tied to confirmation in volume, earnings follow-through, and catalyst delivery. If those validation signals weaken, the build pace should slow immediately and the bullish thesis should be re-tested rather than assumed.",
        PortfolioRating.OVERWEIGHT: "The case supports adding exposure, but only in a measured way, because the upside thesis is still stronger than the downside case while volatility and timing risk remain real. Adds should therefore be linked to catalyst follow-through, valuation digestion, and price support rather than pure momentum chasing. If earnings delivery slips, catalysts fade, or price structure deteriorates, the right move is to pause the adds and move back toward neutral exposure.",
        PortfolioRating.HOLD: "The disciplined move is to maintain current exposure because the upside case is not yet strong enough to justify adding, while the downside case is not yet severe enough to force an immediate trim. The key is to keep monitoring support levels, volume behavior, valuation digestion, and the next round of fundamental confirmation. If those inputs improve together, the setup can be revisited for an add; if support breaks or fundamentals worsen, the stance should shift toward reduction.",
        PortfolioRating.UNDERWEIGHT: "The core logic is to reduce net exposure before the market finishes repricing the current risks, rather than waiting passively for volatility to do the damage. Execution should prioritize trimming the weakest-conviction exposure first while keeping only the most defensible core positions under review. Rebuilding should happen only after valuation resets, catalysts re-accelerate, and price structure stabilizes, not before.",
        PortfolioRating.SELL: "The current risk-reward is unfavorable enough that staying in the name does not offer a justified payoff for the downside being assumed. Execution should therefore prioritize exiting or staying out until both fundamentals and price structure show evidence of repair. Before that happens, attempts to buy the dip would amount to taking uncompensated risk rather than following a disciplined process.",
    }
    return mapping[rating]


def _default_trading_thesis(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "当前上行逻辑更完整，适合在确认信号仍然有效的前提下逐步建立仓位。",
            PortfolioRating.OVERWEIGHT: "当前多头逻辑仍占优，但应以控制节奏的方式增配而不是一次性放大仓位。",
            PortfolioRating.HOLD: "当前多空因素并存，短期缺乏足够的赔率与胜率优势，更适合等待更清晰的确认信号。",
            PortfolioRating.UNDERWEIGHT: "当前风险释放节奏快于新增催化兑现速度，更适合先降低敞口并等待更稳健的再介入条件。",
            PortfolioRating.SELL: "当前风险收益比明显失衡，应以退出仓位或回避参与为主，等待风险重新定价完成。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "The upside thesis is more complete right now, so the setup favors staged accumulation while confirmation remains intact.",
        PortfolioRating.OVERWEIGHT: "The bullish setup still has the edge, but exposure should be increased in a controlled way rather than all at once.",
        PortfolioRating.HOLD: "Bullish and bearish factors are still balanced enough that the setup lacks a clear edge, so waiting is more appropriate.",
        PortfolioRating.UNDERWEIGHT: "Risk is repricing faster than new upside catalysts are materializing, so trimming exposure is more appropriate.",
        PortfolioRating.SELL: "The current risk-reward is unfavorable enough that exiting or staying out is the cleaner choice until repricing runs its course.",
    }
    return mapping[rating]


def _default_execution_plan(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "先以计划目标仓位的20%—30%建立试探仓，只有当价格重新站稳市场分析中给出的首个关键阻力/支撑转换位，且成交量至少较近5日均量放大15%—20%并连续两个交易时段维持，才继续分批加仓。若核心催化只是消息预期而未带来订单、业绩指引或放量突破的确认，就暂停追价，等待回踩确认后再执行下一笔。",
            PortfolioRating.OVERWEIGHT: "在保留现有底仓的前提下择机增配，但每一笔加仓都要绑定清晰的确认条件：价格需要守住市场报告中反复验证的支撑区，成交量至少恢复到20日均量附近，且新增催化要从“预期”变成“可验证进展”。若量价配合不足或催化兑现延迟，就把加仓节奏放慢，只保留已经验证过的核心仓位。",
            PortfolioRating.HOLD: "维持当前仓位，不主动追涨或杀跌。这里的关键支撑应优先参考市场分析中反复出现的50日均线、布林中轨、前低或密集成交区；只有当价格在这些位置附近连续2个交易时段止跌企稳，且成交量至少较近5日均量放大15%—20%或明显回到20日均量上方，才视为成交量改善，可以考虑把持有转为试探性加仓。若新增催化只是消息层面的预期而未带来订单、业绩指引或放量突破确认，则继续维持仓位，不提前放大敞口。",
            PortfolioRating.UNDERWEIGHT: "优先分两到三笔降低敞口，第一笔先处理高弹性但验证最弱的仓位，剩余仓位则观察价格是否在关键支撑附近出现缩量反弹。只有当风险收益比明显改善、价格重新站回主要均线或催化出现实质落地时，才考虑小比例回补；若反弹过程中成交量不足或冲高回落，则继续执行减仓而不是抄底。",
            PortfolioRating.SELL: "以退出仓位或避免入场为主，执行上不要等待模糊修复信号。若价格已跌破主要支撑并伴随放量，就应直接完成清仓；即便后续出现技术性反弹，也要先看到基本面修复、量价结构重建和催化兑现三者同时出现，才考虑重新纳入观察名单，而不是过早回补。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "Start with only 20%-30% of the intended target size and add only after price reclaims the first key support/resistance pivot from the market report, volume runs at least 15%-20% above the recent 5-day average, and the catalyst moves from rumor to verifiable progress. If the move lacks volume or catalyst confirmation, pause the build and wait for a cleaner retest.",
        PortfolioRating.OVERWEIGHT: "Add selectively from the core position, but tie each add to explicit confirmation: price must keep holding the key support zone, volume should recover toward or above the 20-day average, and the catalyst must show measurable follow-through rather than narrative alone. If that confirmation fades, slow the add pace and keep only the already-validated core exposure.",
        PortfolioRating.HOLD: "Maintain current exposure and avoid forcing new trades. Treat the key support zone as the 50-day moving average, Bollinger mid-band, prior swing low, or other repeated support area from the market report; only reconsider adding if price stabilizes there for two trading sessions and volume improves by roughly 15%-20% versus the recent 5-day average or clearly recovers above the 20-day average. If the catalyst remains only a headline without order, guidance, or breakout confirmation, keep the position unchanged.",
        PortfolioRating.UNDERWEIGHT: "Trim exposure in two or three steps, starting with the weakest-conviction slice, and only consider a small rebuild if price reclaims major averages, risk-reward resets, and catalysts regain measurable traction. If rebounds are weak or volume stays thin, keep trimming rather than trying to catch the turn too early.",
        PortfolioRating.SELL: "Prioritize exiting or staying out without waiting for vague repair signals. If price has already broken core support on heavy volume, complete the exit; only revisit the name after fundamentals, catalysts, and price structure all show repair together.",
    }
    return mapping[rating]


def _default_risk_management(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "把失效条件写清楚：若价格重新跌回关键支撑下方，且单日成交量放大到20日均量的1.3倍以上，说明承接失败，应立即暂停加仓并把仓位降回试探水平。同时持续跟踪催化兑现时点、业绩验证与行业相对强弱，避免在只有情绪而没有基本面确认时继续追价。",
            PortfolioRating.OVERWEIGHT: "在增配过程中把单一标的仓位上限、每次加仓比例和失效条件同步约束。若价格连续两天跌破关键支撑，或成交量恢复但股价仍无法突破前高，说明筹码承接不足，应停止加仓并回到核心底仓；若催化兑现不及预期，也要主动降低仓位节奏。",
            PortfolioRating.HOLD: "持续跟踪关键支撑/阻力、成交量与后续业绩指引，并把动作条件写明确：若价格有效跌破关键支撑且单日放量达到20日均量的1.3倍以上，先减掉20%—30%的试探仓位；若价格守住支撑并连续2个交易时段放量修复，再考虑恢复到原仓位。对“成交量改善”的判断不能只看单日放量，至少要结合5日均量、20日均量和价格是否同步收复关键位一起确认。",
            PortfolioRating.UNDERWEIGHT: "在减仓过程中重点看反弹强度、成交量结构和事件兑现进度，避免把缩量反弹误判为趋势修复。若价格反弹但成交量明显弱于20日均量，或催化仍停留在预期阶段，就维持减仓节奏；只有在量价和基本面同步改善时，才允许小比例回补。",
            PortfolioRating.SELL: "在退出或回避期间继续观察是否出现基本面修复与价格结构重建，但不要把短线反弹当作重新入场信号。只有当关键支撑重新站回、成交量恢复到20日均量以上、并且后续催化或业绩验证同步改善时，才考虑重新评估；否则保持空仓或极轻仓观察。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "Define the invalidation clearly: if price falls back below key support and daily volume expands beyond roughly 1.3x the 20-day average, treat that as a failed setup, stop adding, and cut exposure back to probe size. Keep tracking catalyst timing, earnings confirmation, and relative strength so momentum alone does not justify more risk.",
        PortfolioRating.OVERWEIGHT: "While adding, cap single-name exposure, define each add size, and keep explicit failure conditions. If price loses support for two sessions or volume returns without a clean breakout, stop adding and revert to the core position; if catalyst follow-through weakens, slow the sizing pace immediately.",
        PortfolioRating.HOLD: "Track support/resistance, volume, and upcoming guidance with action thresholds attached: if price breaks key support on roughly 1.3x the 20-day average volume, trim 20%-30% of the probing risk; if price stabilizes and reclaims the level with improving volume for two sessions, restore the prior size. Do not call it volume improvement from one noisy session alone—confirm it against both the 5-day and 20-day averages and against price recovery.",
        PortfolioRating.UNDERWEIGHT: "As exposure is reduced, focus on rebound quality, volume structure, and catalyst follow-through so weak countertrend moves are not mistaken for a true repair. Only allow a small rebuild if price, volume, and fundamentals all improve together.",
        PortfolioRating.SELL: "While staying out, keep watching for real repair in fundamentals and price structure, but do not treat a short squeeze or reflex bounce as enough. Reconsider only after support is reclaimed, volume recovers above the 20-day average, and catalysts or guidance improve together.",
    }
    return mapping[rating]


def _default_snapshot_new_and_rebuttal(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "本轮进一步强化了上行催化、盈利兑现与价格承接之间的对应关系，并回应了对估值与波动的主要质疑。",
            PortfolioRating.OVERWEIGHT: "本轮进一步强化了增配逻辑与风险边界之间的对应关系，并回应了对节奏与仓位控制的主要质疑。",
            PortfolioRating.HOLD: "本轮进一步明确了多空证据仍在拉锯，并回应了对过早加仓或过早减仓的主要质疑。",
            PortfolioRating.UNDERWEIGHT: "本轮进一步强化了风险释放节奏与估值约束之间的对应关系，并回应了对过度乐观假设的主要质疑。",
            PortfolioRating.SELL: "本轮进一步强化了退出逻辑与风险收益失衡之间的对应关系，并回应了对继续持仓的主要质疑。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "This round further linked upside catalysts, earnings follow-through, and price support, while addressing the main valuation and volatility objections.",
        PortfolioRating.OVERWEIGHT: "This round clarified why the add-on setup still works within defined risk boundaries and addressed the main pacing objections.",
        PortfolioRating.HOLD: "This round reinforced that the evidence remains mixed and addressed the main objections to changing exposure too early.",
        PortfolioRating.UNDERWEIGHT: "This round clarified the interaction between risk-release cadence and valuation pressure, while addressing the main overly bullish assumptions.",
        PortfolioRating.SELL: "This round reinforced the exit case by linking downside risks to the deteriorating risk-reward profile and addressing the main hold-the-line objections.",
    }
    return mapping[rating]


def _default_snapshot_to_verify(rating: PortfolioRating) -> str:
    if _is_chinese_output():
        mapping = {
            PortfolioRating.BUY: "继续跟踪关键催化、成交量与盈利兑现是否同步验证当前建仓逻辑。",
            PortfolioRating.OVERWEIGHT: "继续跟踪催化兑现、估值消化与价格结构，确认增配逻辑是否继续成立。",
            PortfolioRating.HOLD: "继续跟踪关键支撑/阻力、成交量与后续业绩验证，确认是否出现足以调整仓位的新信号。",
            PortfolioRating.UNDERWEIGHT: "继续跟踪风险释放节奏、关键支撑与估值回落情况，确认是否仍需维持偏谨慎仓位。",
            PortfolioRating.SELL: "继续跟踪基本面修复与价格结构重建信号，确认是否具备重新评估入场的条件。",
        }
        return mapping[rating]
    mapping = {
        PortfolioRating.BUY: "Keep tracking catalysts, volume, and earnings follow-through to confirm that staged accumulation still makes sense.",
        PortfolioRating.OVERWEIGHT: "Keep tracking catalyst delivery, valuation digestion, and price structure to confirm that the add-on case still holds.",
        PortfolioRating.HOLD: "Keep tracking support/resistance, volume, and earnings follow-through to see whether a clearer exposure signal emerges.",
        PortfolioRating.UNDERWEIGHT: "Keep tracking risk-release cadence, key support levels, and valuation reset progress to confirm whether caution is still warranted.",
        PortfolioRating.SELL: "Keep tracking fundamental repair and price stabilization before reconsidering whether re-entry conditions exist.",
    }
    return mapping[rating]


def _sanitize_section(
    text: str,
    default_text: str,
    rating: PortfolioRating,
    *,
    check_action_conflict: bool = False,
    require_detail: bool = False,
) -> str:
    content = (text or "").strip()
    if _is_placeholder_like(content):
        return default_text
    if check_action_conflict and _has_conflicting_primary_action(content, rating):
        return default_text
    if require_detail and _section_needs_detail(content):
        return _merge_sparse_section_with_default(content, default_text)
    return content


def _section_needs_detail(text: str) -> bool:
    content = (text or "").strip()
    if not content:
        return True
    compact = _compact_text(content)
    sentence_count = len(re.findall(r"[。！？!?\.]+", content))
    if _is_chinese_output():
        return len(compact) < 55 or sentence_count < 2
    word_count = len(re.findall(r"\b\w+\b", content))
    return word_count < 18 or sentence_count < 2


def _merge_sparse_section_with_default(content: str, default_text: str) -> str:
    stripped = (content or "").strip()
    if not stripped:
        return default_text
    if _compact_text(stripped) == _compact_text(default_text):
        return default_text
    if _is_chinese_output():
        joiner = "" if stripped.endswith(("。", "！", "？")) else "。"
        return f"{stripped}{joiner}{default_text}"
    joiner = "" if stripped.endswith((".", "!", "?")) else "."
    return f"{stripped}{joiner} {default_text}"


def _sanitize_snapshot_stance(stance: str, rating: PortfolioRating) -> str:
    expected = _expected_rating_text(rating)
    content = (stance or "").strip()
    if not content:
        return expected
    if _is_chinese_output():
        detected = detect_chinese_rating(content)
    else:
        detected = detect_english_rating(content)
    if _contains_any_rating_term(content) and detected != expected:
        return expected
    return content


def render_research_plan(plan: ResearchPlan) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    debate_conclusion = _sanitize_section(
        plan.debate_conclusion,
        _default_debate_conclusion(plan.rating),
        plan.rating,
        require_detail=True,
    )
    action_logic = _sanitize_section(
        plan.action_logic,
        _default_action_logic(plan.rating),
        plan.rating,
        check_action_conflict=True,
        require_detail=True,
    )
    positioning_recommendation = _sanitize_positioning_recommendation(
        plan.positioning_recommendation, plan.rating
    )
    body = (
        f"## {localize_label('Debate Conclusion', '辩论结论')}\n"
        f"{debate_conclusion}\n\n"
        f"## {localize_label('Action Logic', '行为逻辑')}\n"
        f"{action_logic}\n\n"
        f"## {localize_label('Positioning Recommendation', '持仓建议')}\n"
        f"{localize_label('Recommendation', '建议评级')}: {recommendation}\n"
        f"{positioning_recommendation}"
    )
    snapshot = _render_snapshot(
        _sanitize_snapshot_stance(plan.snapshot_stance, plan.rating),
        _sanitize_section(
            plan.snapshot_new_and_rebuttal,
            _default_snapshot_new_and_rebuttal(plan.rating),
            plan.rating,
        ),
        _sanitize_section(
            plan.snapshot_to_verify,
            _default_snapshot_to_verify(plan.rating),
            plan.rating,
        ),
    )
    return f"{body}\n\n{snapshot}".strip()


def render_trader_proposal(plan: TraderProposal) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    thesis = _sanitize_section(
        plan.thesis,
        _default_trading_thesis(plan.rating),
        plan.rating,
        require_detail=True,
    )
    execution_plan = _sanitize_section(
        plan.execution_plan,
        _default_execution_plan(plan.rating),
        plan.rating,
        check_action_conflict=True,
        require_detail=True,
    )
    risk_management = _sanitize_section(
        plan.risk_management,
        _default_risk_management(plan.rating),
        plan.rating,
        require_detail=True,
    )
    if _is_chinese_output():
        return (
            "## 交易逻辑\n"
            f"{thesis}\n\n"
            "## 执行计划\n"
            f"{execution_plan}\n\n"
            "## 风险控制\n"
            f"{risk_management}\n\n"
            f"最终交易建议: **{recommendation}**"
        ).strip()
    return (
        "## Trading Thesis\n"
        f"{thesis}\n\n"
        "## Execution Plan\n"
        f"{execution_plan}\n\n"
        "## Risk Management\n"
        f"{risk_management}\n\n"
        f"FINAL TRANSACTION PROPOSAL: **{recommendation.upper()}**"
    ).strip()


def render_portfolio_decision(plan: PortfolioDecision) -> str:
    recommendation = localize_rating_term(plan.rating.value)
    debate_conclusion = _sanitize_section(
        plan.debate_conclusion,
        _default_debate_conclusion(plan.rating),
        plan.rating,
        require_detail=True,
    )
    action_logic = _sanitize_section(
        plan.action_logic,
        _default_action_logic(plan.rating),
        plan.rating,
        check_action_conflict=True,
        require_detail=True,
    )
    positioning_recommendation = _sanitize_positioning_recommendation(
        plan.positioning_recommendation, plan.rating
    )
    if _is_chinese_output():
        final_line = f"最终交易建议: **{recommendation}**"
    else:
        final_line = f"FINAL TRANSACTION PROPOSAL: **{recommendation.upper()}**"

    body = (
        f"## {localize_label('Debate Conclusion', '辩论结论')}\n"
        f"{debate_conclusion}\n\n"
        f"## {localize_label('Action Logic', '行为逻辑')}\n"
        f"{action_logic}\n\n"
        f"## {localize_label('Positioning Recommendation', '持仓建议')}\n"
        f"{localize_label('Recommendation', '建议评级')}: {recommendation}\n"
        f"{positioning_recommendation}\n\n"
        f"{final_line}"
    )
    snapshot = _render_snapshot(
        _sanitize_snapshot_stance(plan.snapshot_stance, plan.rating),
        _sanitize_section(
            plan.snapshot_new_and_rebuttal,
            _default_snapshot_new_and_rebuttal(plan.rating),
            plan.rating,
        ),
        _sanitize_section(
            plan.snapshot_to_verify,
            _default_snapshot_to_verify(plan.rating),
            plan.rating,
        ),
    )
    return f"{body}\n\n{snapshot}".strip()
