from langchain_core.messages import HumanMessage, RemoveMessage
from difflib import SequenceMatcher
import os
import re

from tradingagents.content_utils import extract_text_content

# Re-export data tools for backward compatibility (analysts import them from here)
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.technical_indicators_tools import get_indicators
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def get_no_greeting_instruction() -> str:
    """Return a prompt instruction forbidding greeting openers."""
    if _is_chinese_output():
        return (
            " 直接进入正文，严禁以问候语开头（如「你好，空头分析师」、「大家好」等），"
            "也不要在正文中重复自己的角色名称。"
        )
    return (
        " Start directly with your argument. Do not open with greetings or salutations "
        "(e.g. 'Hello, Bear Analyst'). Do not repeat your own role name in the body."
    )


def get_output_language() -> str:
    from tradingagents.dataflows.config import get_config

    return get_config().get("output_language", "English")


def _is_chinese_output() -> bool:
    return get_output_language().strip().lower() in {"chinese", "中文", "zh", "zh-cn", "zh-hans"}


def truncate_for_prompt(
    text: str,
    limit_key: str = "report_context_char_limit",
    default_limit: int = 16000,
) -> str:
    """Trim long context text to keep prompts within a stable token budget."""
    if not text:
        return ""

    from tradingagents.dataflows.config import get_config

    cfg = get_config()
    limit = int(cfg.get(limit_key, default_limit) or default_limit)
    if limit <= 0 or len(text) <= limit:
        return text

    omitted = len(text) - limit
    return f"[Content trimmed, omitted {omitted} characters]\n{text[-limit:]}"


def truncate_response_for_prompt(text: str) -> str:
    """Truncate a previous agent argument/response for inclusion as debate context.

    Uses a tighter limit than full reports to prevent debate history from bloating
    the prompt beyond the local model's context window.
    """
    return truncate_for_prompt(text, limit_key="response_context_char_limit", default_limit=3000)


def get_snapshot_template(round_index: int = 1) -> str:
    if _is_chinese_output():
        return """反馈快照:
- 立场:
- 本轮新增与反驳:
- 待验证:"""

    return """FEEDBACK SNAPSHOT:
- Stance:
- New this round & rebuttal:
- To verify:"""


def get_snapshot_writing_instruction(round_index: int = 1) -> str:
    if _is_chinese_output():
        return (
            "反馈快照是对本轮核心内容的完整、详细记录，必须用自己的语言概括，禁止从正文中直接复制句子。\n"
            "每个字段应展开2-4句，包含完整的逻辑链和具体数据，让读者无需阅读正文即可理解本轮论点的全貌：\n\n"
            "「立场」：明确评级 + 核心理由 + 当前最关键的支撑或风险因素。"
            "如「维持减持——30倍PE在商品周期顶部严重透支，MACD死叉确认趋势转弱，"
            "36元支撑位一旦击穿将触发止损盘连锁卖出，下行目标看至32元」；\n\n"
            "「本轮新增与反驳」：把本轮新增的信息增量与它对对手观点的冲击合并写成一个连贯段落。"
            "先写本轮首次提出的新证据、新逻辑或新数据，再明确说明它如何削弱对手【本轮最新主张】。"
            "若本轮引入了新数据，必须解释该数据的含义、为何改变判断，以及它为什么让对手推论难以成立；"
            "若本轮没有新增数据，也要写出本轮最关键的增量判断，并顺带点明它反驳了对手的哪一点；\n\n"
            "「待验证」：列出2-3个下轮需跟踪的具体指标或事件，说明每个指标的阈值和触发含义。"
            "如「①金价能否守稳4800美元——若跌破则确认地缘溢价消退，黄金收益贡献将从30%缩水至15%；"
            "②Q2铜价走势——若跌破8000美元/吨则62%利润增速将面临均值回归压力；"
            "③6月美联储议息——若维持高利率则紫金美元债务成本上升0.3-0.5个百分点」；\n\n"
            "【关键约束】：将你的上轮快照与本轮快照逐字段对比，确保「本轮新增与反驳」"
            "体现了本轮独有的信息增量，而不是复制正文或机械重复上轮表述。\n"
            "严禁开场白，严禁重复角色名，三项内容各不相同。"
        )
    return (
        "The feedback snapshot is a detailed, well-supported record of this round's key content. "
        "Each field should be 2-4 sentences with complete logic chain and specific data, "
        "so the reader can understand the full picture without reading the main argument:\n\n"
        "'Stance': rating + core rationale + the single most critical supporting or risk factor right now. "
        "e.g. 'Maintain Sell — 30x PE at the top of a commodity cycle is severely stretched; "
        "MACD death cross confirms trend weakening; if $36 support breaks, stop-loss cascade targets $32';\n\n"
        "'New this round & rebuttal': merge the incremental content and the rebuttal into one coherent field. "
        "First state the genuinely new evidence, framing, or data introduced this round, then explain how it weakens "
        "the opponent's latest claim. If a new metric appears, explain why it matters and why it undermines the opposing inference;\n\n"
        "'To verify': 2-3 specific indicators or events with thresholds and trigger meanings. "
        "e.g. '① Copper holding above $8,000 — if it breaks, 62% profit growth faces mean-reversion; "
        "② Fed June decision — if rates held, USD debt cost rises 30-50bps; "
        "③ Q2 earnings capex — if above $5B, confirms sustainable expansion thesis'.\n\n"
        "KEY CONSTRAINT: Compare previous snapshot with this one field by field — "
        "'New this round & rebuttal' must contain real incremental content instead of copied body text or repeated phrasing.\n"
        "No greetings. No role names. No two fields alike."
    )

def localize_label(english: str, chinese: str) -> str:
    return chinese if _is_chinese_output() else english


def localize_role_name(role: str) -> str:
    mapping = {
        "Bull Analyst": "多头分析师",
        "Bear Analyst": "空头分析师",
        "Aggressive Analyst": "激进分析师",
        "Conservative Analyst": "保守分析师",
        "Neutral Analyst": "中性分析师",
        "Portfolio Manager": "投资组合经理",
        "Research Manager": "研究经理",
        "Trader": "交易员",
        "Judge": "裁决者",
    }
    return mapping.get(role, role) if _is_chinese_output() else role


# Reverse mapping: Chinese → English (for always-include-both logic)
_ROLE_BOTH_NAMES: dict[str, set[str]] = {
    en: {en, zh}
    for en, zh in {
        "Bull Analyst": "多头分析师",
        "Bear Analyst": "空头分析师",
        "Aggressive Analyst": "激进分析师",
        "Conservative Analyst": "保守分析师",
        "Neutral Analyst": "中性分析师",
        "Portfolio Manager": "投资组合经理",
        "Research Manager": "研究经理",
        "Trader": "交易员",
        "Judge": "裁决者",
    }.items()
}


def normalize_chinese_role_terms(text: str) -> str:
    """Normalize user-facing Chinese role terms to a single preferred wording."""
    if not text:
        return ""
    return CHINESE_ROLE_TERM_PATTERN.sub(
        lambda match: CHINESE_ROLE_TERM_REPLACEMENTS[match.group(0)],
        text,
    )


def localize_rating_term(term: str) -> str:
    mapping = {
        "Buy": "买入",
        "Overweight": "增持",
        "Hold": "持有",
        "Underweight": "减持",
        "Sell": "卖出",
        "BUY": "买入",
        "HOLD": "持有",
        "SELL": "卖出",
    }
    return mapping.get(term, term) if _is_chinese_output() else term


def get_localized_rating_scale() -> str:
    if _is_chinese_output():
        return """**评级体系**（只能选择一个）:
- **买入**: 对开仓或加仓有很强信心
- **增持**: 前景偏积极，建议逐步提高仓位
- **持有**: 维持当前仓位，暂不动作
- **减持**: 降低敞口，分批止盈或收缩仓位
- **卖出**: 退出仓位或避免入场"""

    return """**Rating Scale** (use exactly one):
- **Buy**: Strong conviction to enter or add to position
- **Overweight**: Favorable outlook, gradually increase exposure
- **Hold**: Maintain current position, no action needed
- **Underweight**: Reduce exposure, take partial profits
- **Sell**: Exit position or avoid entry"""


def get_localized_final_proposal_instruction() -> str:
    if _is_chinese_output():
        return (
            "以明确的交易建议结束，最后一行必须使用格式："
            "'最终交易建议: **买入/增持/持有/减持/卖出**'。"
        )
    return (
        "End with a firm decision and always conclude your response with "
        "'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation."
    )


def get_bear_proposal_instruction() -> str:
    """Consistency instruction for the Bear/Short Analyst.
    The conclusion must logically follow from the bearish arguments presented.
    """
    if _is_chinese_output():
        return (
            "在给出最终结论前，先自我审查：你的评级是否与你所列举的风险和负面证据一致？"
            "作为空头分析师，如果你的论点强调了下行风险，结论就应反映这一判断（通常为减持或卖出）。"
            "不允许出现论点强调风险、结论却比多头分析师还乐观的情形。"
            "在文末输出“决策摘要”块，并在其中给出“评级: 减持/卖出”（或在有力证据支撑时用“持有”）。"
        )
    return (
        "Before stating your conclusion, verify self-consistency: does your rating logically follow from the risks "
        "and downsides you've argued? As a Bear Analyst, if your argument emphasizes significant risks, your "
        "conclusion should reflect that (typically UNDERWEIGHT or SELL). You must not conclude more optimistically "
        "than your own arguments warrant — a bearish case cannot end with BUY or OVERWEIGHT. "
        "End with a 'DECISION SUMMARY' block whose rating is UNDERWEIGHT or SELL "
        "(or HOLD only when your arguments genuinely support a neutral stance)."
    )


def get_bull_proposal_instruction() -> str:
    """Consistency instruction for the Bull/Long Analyst.
    The conclusion must logically follow from the bullish arguments presented.
    """
    if _is_chinese_output():
        return (
            "在给出最终结论前，先自我审查：你的评级是否与你所列举的增长潜力和正面证据一致？"
            "作为多头分析师，如果你的论点强调了上行机会，结论就应反映这一判断（通常为买入或增持）。"
            "不允许出现论点强调机会、结论却比空头分析师还悲观的情形。"
            "在文末输出“决策摘要”块，并在其中给出“评级: 买入/增持”（或在有力证据支撑时用“持有”）。"
        )
    return (
        "Before stating your conclusion, verify self-consistency: does your rating logically follow from the "
        "growth potential and positive indicators you've argued? As a Bull Analyst, if your argument emphasizes "
        "strong upside, your conclusion should reflect that (typically BUY or OVERWEIGHT). You must not conclude "
        "more pessimistically than your own arguments warrant — a bullish case cannot end with SELL or UNDERWEIGHT. "
        "End with a 'DECISION SUMMARY' block whose rating is BUY or OVERWEIGHT "
        "(or HOLD only when your arguments genuinely support a neutral stance)."
    )


def get_analyst_decision_template() -> str:
    if _is_chinese_output():
        return """决策摘要:
- 评级:
- 置信度:
- 时间区间:
- 关键假设:
  1.
  2.
  3."""
    return """DECISION SUMMARY:
- Rating:
- Confidence:
- Time Horizon:
- Key Assumptions:
  1.
  2.
  3."""


def get_analyst_decision_instruction() -> str:
    if _is_chinese_output():
        return (
            "请在正文结束后、反馈快照之前，追加一个严格使用 Markdown 的“决策摘要”块。"
            "该块必须包含四项：评级、置信度、时间区间、关键假设（3条）。"
            "不要输出 XML、JSON、代码块、<final_answer> 标签，也不要输出英文标题 "
            "“Reflections from similar situations and lessons learned”。"
            "历史经验只允许内部吸收，不得在可见答案中直接引用、复述或翻译。"
        )
    return (
        "After the main argument and before the feedback snapshot, append a strict Markdown "
        "'DECISION SUMMARY' block with exactly four fields: Rating, Confidence, Time Horizon, "
        "and Key Assumptions (3 bullets). Do not output XML, JSON, code fences, or "
        "'<final_answer>' tags. Use past lessons internally only — do not quote or restate them."
    )


def get_aggressive_risk_instruction() -> str:
    """Consistency instruction for the Aggressive Risk Analyst.
    Natural tendency: bullish/aggressive, but any action is allowed if evidence warrants.
    The conclusion uses the dedicated '风险建议' format with pace/scale vocabulary.
    """
    if _is_chinese_output():
        return (
            "先自我审查：你「立场」字段是否与你本轮实际论点一致？"
            "作为激进分析师，你的自然倾向是寻找高回报机会，但若本轮证据确实指向风险，"
            "也可以选择逐步减仓或立即止损——重要的是结论与论点保持自洽，而非机械地执行激进立场。"
            "请在正文结尾处使用格式 '风险建议: **[行动]**'。\n"
            "请从以下完整词汇表中选择最符合你本轮论点的行动，可加修饰词体现节奏、幅度和条件：\n激进加仓 / 坚决买入 / 分批建仓 / 满仓做多 / 小幅加仓 / 持仓不动 /\n维持现仓 / 谨慎观望 / 分批调整 / 设止损后持有 / 小幅减仓 /\n逐步减仓 / 分批卖出 / 谨慎持有 / 坚决减仓 / 立即止损\n示例：风险建议: **逐步减仓，控制回撤至5%以内**  或  风险建议: **分批建仓，首批仓位不超过30%**"
        )
    return (
        "Before appending the snapshot, verify self-consistency: does your 'Stance' field "
        "match what your arguments actually argued this round? "
        "As the Aggressive Analyst your natural lean is high-reward entry, but if evidence "
        "this round clearly points to risk, gradual reduction or a stop-loss is valid — "
        "what matters is that your conclusion is self-consistent, not mechanically aggressive. "
        "Conclude your main argument body with a line using format 'RISK RECOMMENDATION: **[action]**'.\n"
        "Choose the action that best fits your argument this round from the full vocabulary below, adding modifiers for pace, scale, and conditions as needed:\nAggressively Accumulate / Decisively Buy / Build in Batches / Go All-In / Slightly Increase / Hold Firm /\nMaintain Position / Cautiously Observe / Adjust in Batches / Hold with Stop-Loss / Slightly Reduce /\nGradually Reduce / Sell in Batches / Hold with Caution / Decisively Reduce / Immediate Stop-Loss\nExample: RISK RECOMMENDATION: **Gradually Reduce, Cap Drawdown at 5%**  or  RISK RECOMMENDATION: **Build in Batches, Initial Position <=30%**"
    )

def get_conservative_risk_instruction() -> str:
    """Consistency instruction for the Conservative Risk Analyst.
    Natural tendency: cautious/defensive, but any action is allowed if evidence warrants.
    The conclusion uses the dedicated '风险建议' format with pace/scale vocabulary.
    """
    if _is_chinese_output():
        return (
            "先自我审查：你「立场」字段是否与你本轮实际论点一致？"
            "作为保守分析师，你的自然倾向是保护资产、控制回撤，但若本轮证据确实支持机会，"
            "也可以选择维持仓位甚至小幅加仓——重要的是结论与论点保持自洽，而非机械地执行保守立场。"
            "请在正文结尾处使用格式 '风险建议: **[行动]**'。\n"
            "请从以下完整词汇表中选择最符合你本轮论点的行动，可加修饰词体现节奏、幅度和条件：\n激进加仓 / 坚决买入 / 分批建仓 / 满仓做多 / 小幅加仓 / 持仓不动 /\n维持现仓 / 谨慎观望 / 分批调整 / 设止损后持有 / 小幅减仓 /\n逐步减仓 / 分批卖出 / 谨慎持有 / 坚决减仓 / 立即止损\n示例：风险建议: **逐步减仓，控制回撤至5%以内**  或  风险建议: **分批建仓，首批仓位不超过30%**"
        )
    return (
        "Before appending the snapshot, verify self-consistency: does your 'Stance' field "
        "match what your arguments actually argued this round? "
        "As the Conservative Analyst your natural lean is capital protection, but if evidence "
        "this round genuinely supports a position, holding or a small increase is valid — "
        "what matters is that your conclusion is self-consistent, not mechanically cautious. "
        "Conclude your main argument body with a line using format 'RISK RECOMMENDATION: **[action]**'.\n"
        "Choose the action that best fits your argument this round from the full vocabulary below, adding modifiers for pace, scale, and conditions as needed:\nAggressively Accumulate / Decisively Buy / Build in Batches / Go All-In / Slightly Increase / Hold Firm /\nMaintain Position / Cautiously Observe / Adjust in Batches / Hold with Stop-Loss / Slightly Reduce /\nGradually Reduce / Sell in Batches / Hold with Caution / Decisively Reduce / Immediate Stop-Loss\nExample: RISK RECOMMENDATION: **Gradually Reduce, Cap Drawdown at 5%**  or  RISK RECOMMENDATION: **Build in Batches, Initial Position <=30%**"
    )

def get_neutral_risk_instruction() -> str:
    """Consistency instruction for the Neutral Risk Analyst.
    Natural tendency: balanced, but any action is allowed if evidence clearly leans one way.
    The conclusion uses the dedicated '风险建议' format with pace/scale vocabulary.
    """
    if _is_chinese_output():
        return (
            "先自我审查：你「立场」字段是否与你本轮实际论点一致？"
            "作为中性分析师，你的自然倾向是平衡两方观点，但若本轮证据明确偏向某一侧，"
            "也可以选择激进加仓或坚决减仓——重要的是结论与论点保持自洽，而非机械地居中立场。"
            "请在正文结尾处使用格式 '风险建议: **[行动]**'。\n"
            "请从以下完整词汇表中选择最符合你本轮论点的行动，可加修饰词体现节奏、幅度和条件：\n激进加仓 / 坚决买入 / 分批建仓 / 满仓做多 / 小幅加仓 / 持仓不动 /\n维持现仓 / 谨慎观望 / 分批调整 / 设止损后持有 / 小幅减仓 /\n逐步减仓 / 分批卖出 / 谨慎持有 / 坚决减仓 / 立即止损\n示例：风险建议: **逐步减仓，控制回撤至5%以内**  或  风险建议: **分批建仓，首批仓位不超过30%**"
        )
    return (
        "Before appending the snapshot, verify self-consistency: does your 'Stance' field "
        "match what your arguments actually argued this round? "
        "As the Neutral Analyst your natural lean is balance, but if evidence "
        "this round clearly favors one side, aggressive accumulation or decisive reduction is valid — "
        "what matters is that your conclusion is self-consistent, not mechanically neutral. "
        "Conclude your main argument body with a line using format 'RISK RECOMMENDATION: **[action]**'.\n"
        "Choose the action that best fits your argument this round from the full vocabulary below, adding modifiers for pace, scale, and conditions as needed:\nAggressively Accumulate / Decisively Buy / Build in Batches / Go All-In / Slightly Increase / Hold Firm /\nMaintain Position / Cautiously Observe / Adjust in Batches / Hold with Stop-Loss / Slightly Reduce /\nGradually Reduce / Sell in Batches / Hold with Caution / Decisively Reduce / Immediate Stop-Loss\nExample: RISK RECOMMENDATION: **Gradually Reduce, Cap Drawdown at 5%**  or  RISK RECOMMENDATION: **Build in Batches, Initial Position <=30%**"
    )



def get_collaboration_stop_instruction() -> str:
    if _is_chinese_output():
        return (
            " 如果你或其他助手已经给出了最终结论，请在响应中包含："
            "'最终交易建议: **买入/增持/持有/减持/卖出**'，团队将以此为停止信号。"
        )
    return (
        " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
        " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
    )


SNAPSHOT_MARKERS = ("FEEDBACK SNAPSHOT:", "反馈快照:")
SNAPSHOT_TEMPLATE = get_snapshot_template()
DECISION_SUMMARY_MARKERS = ("DECISION SUMMARY:", "决策摘要:")
_XML_FINAL_ANSWER_RE = re.compile(
    r"```xml\s*(<final_answer>.*?</final_answer>)\s*```|(<final_answer>.*?</final_answer>)",
    re.IGNORECASE | re.DOTALL,
)
_REFLECTIONS_HEADING_RE = re.compile(
    r"\*{0,2}Reflections from similar situations and lessons learned:?\*{0,2}",
    re.IGNORECASE,
)
_RISK_RECOMMENDATION_RE = re.compile(
    r"(?:风险建议|RISK RECOMMENDATION)\s*[:：]\s*\**(.+?)\**(?:$|\n)",
    re.IGNORECASE,
)
CHINESE_RATING_EXPLICIT_PATTERNS = [
    ("买入", re.compile(r"(?:最终交易建议|评级)\s*[:：]\s*\**买入\**")),
    ("增持", re.compile(r"(?:最终交易建议|评级)\s*[:：]\s*\**增持\**")),
    ("持有", re.compile(r"(?:最终交易建议|评级)\s*[:：]\s*\**持有\**")),
    ("减持", re.compile(r"(?:最终交易建议|评级)\s*[:：]\s*\**减持\**")),
    ("卖出", re.compile(r"(?:最终交易建议|评级)\s*[:：]\s*\**卖出\**")),
    ("买入", re.compile(r"(?:立场|当前观点)\s*[:：]\s*\**买入(?:/增持)?\**")),
    ("增持", re.compile(r"(?:立场|当前观点)\s*[:：]\s*\**增持(?:/买入)?\**")),
    ("持有", re.compile(r"(?:立场|当前观点)\s*[:：]\s*\**持有\**")),
    ("减持", re.compile(r"(?:立场|当前观点)\s*[:：]\s*\**减持\**")),
    ("卖出", re.compile(r"(?:立场|当前观点)\s*[:：]\s*\**卖出\**")),
    ("买入", re.compile(r"(?:建议|维持|转为)\s*买入")),
    ("增持", re.compile(r"(?:建议|维持|转为)\s*增持")),
    ("持有", re.compile(r"(?:建议|维持|转为)\s*持有")),
    ("减持", re.compile(r"(?:建议|维持|转为)\s*减持")),
    ("卖出", re.compile(r"(?:建议|维持|转为)\s*卖出")),
]
CHINESE_RATING_NEGATION_PATTERN = re.compile(
    r"(不建议|不宜|不是|并非|避免|不要|勿|难言|无法|不能|非|不|别)\s*$"
)
CHINESE_RATING_HEURISTIC_PATTERNS = [
    ("卖出", re.compile(r"(清仓|退出(?:仓位|头寸)?|止损离场|果断卖出|卖出为主)")),
    ("减持", re.compile(r"(降低仓位|分批止盈|降低敞口|部分卖出|先减仓|逢高减仓|止盈减仓)")),
    ("增持", re.compile(r"(加仓|提高仓位|逢低布局|继续增持|扩大仓位)")),
    ("买入", re.compile(r"(买入机会|积极布局|值得买入|坚定看多|继续买入)")),
    ("持有", re.compile(r"(继续观察|暂不动作|维持仓位|等待确认|持仓观望)")),
]
ENGLISH_RATING_EXPLICIT_PATTERNS = [
    ("SELL", re.compile(r"(?:final transaction proposal|rating)\s*:\s*\**sell\**", re.IGNORECASE)),
    ("UNDERWEIGHT", re.compile(r"(?:final transaction proposal|rating)\s*:\s*\**underweight\**", re.IGNORECASE)),
    ("HOLD", re.compile(r"(?:final transaction proposal|rating)\s*:\s*\**hold\**", re.IGNORECASE)),
    ("OVERWEIGHT", re.compile(r"(?:final transaction proposal|rating)\s*:\s*\**overweight\**", re.IGNORECASE)),
    ("BUY", re.compile(r"(?:final transaction proposal|rating)\s*:\s*\**buy\**", re.IGNORECASE)),
    ("BUY", re.compile(r"(?:stance|current thesis)\s*:\s*\**buy(?:/overweight)?\**", re.IGNORECASE)),
    ("OVERWEIGHT", re.compile(r"(?:stance|current thesis)\s*:\s*\**overweight(?:/buy)?\**", re.IGNORECASE)),
    ("HOLD", re.compile(r"(?:stance|current thesis)\s*:\s*\**hold\**", re.IGNORECASE)),
    ("UNDERWEIGHT", re.compile(r"(?:stance|current thesis)\s*:\s*\**underweight\**", re.IGNORECASE)),
    ("SELL", re.compile(r"(?:stance|current thesis)\s*:\s*\**sell\**", re.IGNORECASE)),
    ("SELL", re.compile(r"(?:recommend|maintain|shift to|move to)\s+sell", re.IGNORECASE)),
    ("UNDERWEIGHT", re.compile(r"(?:recommend|maintain|shift to|move to)\s+underweight", re.IGNORECASE)),
    ("HOLD", re.compile(r"(?:recommend|maintain|shift to|move to)\s+hold", re.IGNORECASE)),
    ("OVERWEIGHT", re.compile(r"(?:recommend|maintain|shift to|move to)\s+overweight", re.IGNORECASE)),
    ("BUY", re.compile(r"(?:recommend|maintain|shift to|move to)\s+buy", re.IGNORECASE)),
]
ENGLISH_RATING_NEGATION_PATTERN = re.compile(
    r"(do not|don't|not|avoid|never|no|cannot|can't)\s*$",
    re.IGNORECASE,
)
ENGLISH_RATING_HEURISTIC_PATTERNS = [
    ("SELL", re.compile(r"(exit position|sell the stock|close the position|fully exit)", re.IGNORECASE)),
    ("UNDERWEIGHT", re.compile(r"(reduce exposure|trim the position|take partial profits)", re.IGNORECASE)),
    ("OVERWEIGHT", re.compile(r"(add to position|increase exposure|build the position)", re.IGNORECASE)),
    ("BUY", re.compile(r"(buy the stock|enter the position|strong upside)", re.IGNORECASE)),
    ("HOLD", re.compile(r"(maintain the position|wait for confirmation|stay on hold)", re.IGNORECASE)),
]


def _condense_excerpt(text: str, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _get_rating_patterns() -> list[tuple[str, tuple[str, ...]]]:
    return [
        ("买入", ("最终交易建议: **买入**", "评级: 买入", "建议买入", "维持买入", "转为买入", "买入")),
        ("增持", ("最终交易建议: **增持**", "评级: 增持", "建议增持", "维持增持", "转为增持", "增持")),
        ("持有", ("最终交易建议: **持有**", "评级: 持有", "建议持有", "维持持有", "转为持有", "持有")),
        ("减持", ("最终交易建议: **减持**", "评级: 减持", "建议减持", "维持减持", "转为减持", "减持")),
        ("卖出", ("最终交易建议: **卖出**", "评级: 卖出", "建议卖出", "维持卖出", "转为卖出", "卖出")),
    ]


def _detect_risk_stance(text: str) -> str:
    content = normalize_chinese_role_terms(text or "")
    if not content.strip():
        return ""

    match = _RISK_RECOMMENDATION_RE.search(content)
    if match:
        return match.group(1).strip().strip("*").strip()
    return ""


def _detect_chinese_rating(text: str) -> str:
    content = normalize_chinese_role_terms(text or "")
    if not content.strip():
        return "持有"

    for rating, pattern in CHINESE_RATING_EXPLICIT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        prefix = content[max(0, match.start() - 8):match.start()]
        if CHINESE_RATING_NEGATION_PATTERN.search(prefix):
            continue
        return rating

    for rating, pattern in CHINESE_RATING_HEURISTIC_PATTERNS:
        for match in pattern.finditer(content):
            prefix = content[max(0, match.start() - 8):match.start()]
            if CHINESE_RATING_NEGATION_PATTERN.search(prefix):
                continue
            return rating

    return "持有"


def _detect_english_rating(text: str) -> str:
    content = (text or "").lower()
    if not content.strip():
        return "HOLD"

    for rating, pattern in ENGLISH_RATING_EXPLICIT_PATTERNS:
        match = pattern.search(content)
        if not match:
            continue
        prefix = content[max(0, match.start() - 12):match.start()]
        if ENGLISH_RATING_NEGATION_PATTERN.search(prefix):
            continue
        return rating

    for rating, pattern in ENGLISH_RATING_HEURISTIC_PATTERNS:
        for match in pattern.finditer(content):
            prefix = content[max(0, match.start() - 12):match.start()]
            if ENGLISH_RATING_NEGATION_PATTERN.search(prefix):
                continue
            return rating

    return "HOLD"


def _extract_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    parts = re.split(r"(?<=[。！？!?\.])\s+|\n+", compact)
    return [part.strip() for part in parts if part.strip()]


def _pick_sentence(sentences: list[str], keywords: tuple[str, ...]) -> str:
    for sentence in sentences:
        if any(keyword in sentence for keyword in keywords):
            return _condense_excerpt(sentence, 120)
    return ""


def _strip_snapshot_discourse_openers(text: str) -> str:
    cleaned = _strip_any_role_prefix_from_value(text)
    cleaned = re.sub(r"^(?:各位|大家)[，,、:：!\s]*", "", cleaned)
    cleaned = re.sub(
        r"^(?:首先|其次|再看|另外|此外|最后|总体来看|总的来说|需要注意的是|值得注意的是|更重要的是|具体来说)[，,、:：\s]*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:first|second|third|next|finally|overall|more importantly|in addition|however)\b[,\s:;.-]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _normalize_overlap_text(text: str) -> str:
    lowered = normalize_chinese_role_terms((text or "").lower())
    return re.sub(r"[\W_]+", "", lowered, flags=re.UNICODE)


def _contains_placeholder_snapshot(snapshot: str) -> bool:
    placeholders = (
        "未明确说明",
        "暂无",
        "同上",
        "无变化",
        "Not explicitly stated",
        "None yet",
        "same as above",
        "no change",
    )
    return any(token in snapshot for token in placeholders)


def _snapshot_output_is_chinese(chinese: bool | None = None) -> bool:
    return _is_chinese_output() if chinese is None else chinese


def _snapshot_uses_chinese(snapshot: str) -> bool:
    if SNAPSHOT_MARKERS[1] in snapshot:
        return True
    if SNAPSHOT_MARKERS[0] in snapshot:
        return False
    if re.search(r"- (?:立场|本轮新增与反驳|本轮新增|发生了什么变化|为什么变化|关键反驳|待验证|下一轮教训):", snapshot):
        return True
    if re.search(r"- (?:Stance|Current thesis|New this round & rebuttal|New this round|What changed|Why it changed|Key rebuttal|To verify|Lesson for next round):", snapshot):
        return False
    return _is_chinese_output()


def _snapshot_field_labels(chinese: bool | None = None) -> list[str]:
    if _snapshot_output_is_chinese(chinese):
        return ["立场", "本轮新增与反驳", "待验证"]
    return ["Stance", "New this round & rebuttal", "To verify"]


def _snapshot_field_aliases() -> dict[str, tuple[str, ...]]:
    return {
        # primary label first; legacy labels kept for backward-compatible parsing
        "stance": ("立场", "Stance", "当前观点", "Current thesis"),
        "new_and_rebuttal": (
            "本轮新增与反驳",
            "New this round & rebuttal",
            "本轮新增/反驳",
            "New this round / rebuttal",
            "本轮新增",
            "New this round",
            "发生了什么变化",
            "What changed",
            "为什么变化",
            "Why it changed",
            "关键反驳",
            "Key rebuttal",
        ),
        "to_verify": ("待验证", "To verify", "下一轮教训", "Lesson for next round"),
    }


def _decision_field_labels() -> list[str]:
    if _is_chinese_output():
        return ["评级", "置信度", "时间区间", "关键假设"]
    return ["Rating", "Confidence", "Time Horizon", "Key Assumptions"]


def _decision_field_aliases() -> dict[str, tuple[str, ...]]:
    return {
        "rating": ("评级", "Rating", "结论", "Conclusion"),
        "confidence": ("置信度", "Confidence", "Confidence Level"),
        "time_horizon": ("时间区间", "Time Horizon", "持有周期"),
        "key_assumptions": ("关键假设", "Key Assumptions"),
    }


def _empty_decision_fields() -> dict[str, str]:
    return {key: "" for key in _decision_field_aliases()}


def _parse_markdown_decision_fields(block: str) -> dict[str, str]:
    fields = _empty_decision_fields()
    if not block:
        return fields

    current_key = None
    for raw_line in block.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped in DECISION_SUMMARY_MARKERS:
            continue

        matched = False
        for field_key, aliases in _decision_field_aliases().items():
            for label in aliases:
                prefix = f"- {label}:"
                if stripped.startswith(prefix):
                    fields[field_key] = stripped[len(prefix):].strip()
                    current_key = field_key
                    matched = True
                    break
            if matched:
                break

        if matched:
            continue

        if current_key == "key_assumptions" and re.match(r"^(?:[-*]|\d+\.)\s+", stripped):
            existing = fields[current_key]
            fields[current_key] = (
                existing + ("\n" if existing else "") + stripped
            )
            continue

        if current_key and not stripped.startswith("-"):
            fields[current_key] = (fields[current_key] + " " + stripped).strip()

    return fields


def _parse_xml_decision_fields(text: str) -> dict[str, str]:
    fields = _empty_decision_fields()
    if not text:
        return fields

    match = _XML_FINAL_ANSWER_RE.search(text)
    if not match:
        return fields

    xml = match.group(1) or match.group(2) or ""
    tag_mapping = {
        "rating": ("conclusion",),
        "confidence": ("confidence_level",),
        "time_horizon": ("time_horizon",),
        "key_assumptions": ("key_assumptions",),
    }
    for field_key, tags in tag_mapping.items():
        for tag in tags:
            tag_match = re.search(
                rf"<{tag}>\s*(.*?)\s*</{tag}>",
                xml,
                re.IGNORECASE | re.DOTALL,
            )
            if tag_match:
                fields[field_key] = tag_match.group(1).strip()
                break

    assumptions = []
    if fields["key_assumptions"]:
        for line in fields["key_assumptions"].splitlines():
            cleaned = line.strip()
            if cleaned:
                assumptions.append(cleaned)
        fields["key_assumptions"] = "\n".join(assumptions)

    return fields


def _format_decision_summary_from_fields(fields: dict[str, str]) -> str:
    if not any(value.strip() for value in fields.values()):
        return ""

    title = DECISION_SUMMARY_MARKERS[1] if _is_chinese_output() else DECISION_SUMMARY_MARKERS[0]
    labels = _decision_field_labels()
    keys = list(_decision_field_aliases().keys())
    lines = [title]

    for key, label in zip(keys[:-1], labels[:-1]):
        value = fields.get(key, "").strip()
        if value:
            lines.append(f"- {label}: {value}")

    assumptions = fields.get("key_assumptions", "").strip()
    if assumptions:
        lines.append(f"- {labels[-1]}:")
        for line in assumptions.splitlines():
            lines.append(f"  {line.strip()}")

    return "\n".join(lines)


def extract_analyst_decision_summary(text: str) -> str:
    """Extract a structured analyst decision summary block, normalizing legacy XML."""
    if not text:
        return ""

    best_idx = -1
    best_marker = None
    for marker in DECISION_SUMMARY_MARKERS:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
            best_marker = marker

    if best_idx != -1 and best_marker is not None:
        end = len(text)
        for marker in SNAPSHOT_MARKERS:
            idx = text.find(marker, best_idx)
            if idx != -1:
                end = min(end, idx)
        block = text[best_idx:end].strip()
        fields = _parse_markdown_decision_fields(block)
        normalized = _format_decision_summary_from_fields(fields)
        return normalized or block

    fields = _parse_xml_decision_fields(text)
    return _format_decision_summary_from_fields(fields)


def _strip_reflections_section(text: str) -> str:
    if not text:
        return ""
    match = _REFLECTIONS_HEADING_RE.search(text)
    if not match:
        return text

    start = match.start()
    end = len(text)
    for marker in (*DECISION_SUMMARY_MARKERS, *SNAPSHOT_MARKERS):
        idx = text.find(marker, match.end())
        if idx != -1:
            end = min(end, idx)
    return (text[:start] + text[end:]).strip()


def strip_analyst_decision_summary(text: str) -> str:
    """Remove decision-summary/meta blocks from visible analyst body text."""
    if not text:
        return ""

    cleaned = _strip_reflections_section(text)
    cleaned = _XML_FINAL_ANSWER_RE.sub("", cleaned)

    best_idx = -1
    for marker in DECISION_SUMMARY_MARKERS:
        idx = cleaned.rfind(marker)
        if idx > best_idx:
            best_idx = idx
    if best_idx != -1:
        cleaned = cleaned[:best_idx].rstrip()

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


_ALL_ROLE_NAMES: tuple[str, ...] = (
    "多头分析师", "空头分析师", "激进分析师", "保守分析师", "中性分析师", "投资组合经理",
    "Bull Analyst", "Bear Analyst", "Aggressive Analyst", "Conservative Analyst",
    "Neutral Analyst", "Portfolio Manager",
)


def _strip_any_role_prefix_from_value(value: str) -> str:
    """Strip a leading role self-label and greeting from a snapshot field value.

    e.g. '保守分析师: 各位分析师好，我是保守风险分析师。针对...' → '针对...'
    """
    import re
    pattern = "|".join(re.escape(n) for n in _ALL_ROLE_NAMES)
    # Strip "rolename：/: " at the very start
    value = re.sub(r"^(?:" + pattern + r")[：:]\s*", "", value)
    # Strip common self-introduction sentences at the start:
    # e.g. "各位分析师好，我是保守风险分析师。" / "大家好，我是激进分析师。"
    value = re.sub(
        r"^(?:各位\S{0,6}好[，,。！!]?\s*)?(?:我是|作为)\S{2,12}(?:分析师|经理)[，,。！!。]\s*",
        "",
        value,
    )
    return value.strip()


def _parse_snapshot_fields(snapshot: str) -> dict[str, str]:
    fields = {key: "" for key in _snapshot_field_aliases()}
    if not snapshot:
        return fields

    chinese_snapshot = _snapshot_uses_chinese(snapshot)
    current_key = None
    for line in snapshot.splitlines():
        stripped = line.strip()
        matched = False
        for field_key, aliases in _snapshot_field_aliases().items():
            for label in aliases:
                prefix = f"- {label}:"
                if stripped.startswith(prefix):
                    new_value = stripped[len(prefix):].strip()
                    fields[field_key] = _merge_snapshot_field_values(
                        fields[field_key],
                        new_value,
                        chinese_snapshot,
                    )
                    current_key = field_key
                    matched = True
                    break
            if matched:
                break
        if not matched and current_key and stripped and not stripped.startswith("-"):
            # Continuation line for the current field — join with a space
            fields[current_key] = (fields[current_key] + " " + stripped).strip()

    # Strip role prefixes after full value is assembled
    for key in fields:
        fields[key] = _strip_any_role_prefix_from_value(fields[key])
    return fields


def _format_snapshot_from_fields(fields: dict[str, str], chinese: bool | None = None) -> str:
    title = SNAPSHOT_MARKERS[1] if _snapshot_output_is_chinese(chinese) else SNAPSHOT_MARKERS[0]
    labels = _snapshot_field_labels(chinese)
    lines = [title]
    for field_key, label in zip(_snapshot_field_aliases().keys(), labels):
        value = fields.get(field_key, "").strip()
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _snapshot_has_missing_fields(snapshot: str) -> bool:
    fields = _parse_snapshot_fields(snapshot)
    for value in fields.values():
        normalized = value.strip()
        if not normalized:
            return True
        if normalized in {"。", ".", "...", "……", "-", "--"}:
            return True
    return False


def _merge_snapshot_with_inferred(
    snapshot: str,
    inferred_snapshot: str,
    replacement_fields: set[str] | None = None,
    chinese: bool | None = None,
) -> str:
    explicit = _parse_snapshot_fields(snapshot)
    inferred = _parse_snapshot_fields(inferred_snapshot)
    replacement_fields = replacement_fields or set()

    lines = [SNAPSHOT_MARKERS[1] if _snapshot_output_is_chinese(chinese) else SNAPSHOT_MARKERS[0]]
    display_labels = _snapshot_field_labels(chinese)
    for field_key, label in zip(_snapshot_field_aliases().keys(), display_labels):
        value = explicit.get(field_key, "").strip()
        if (
            not value
            or value in {"。", ".", "...", "……", "-", "--"}
            or field_key in replacement_fields
        ):
            value = inferred.get(field_key, "").strip()
        lines.append(f"- {label}: {value}")
    return "\n".join(lines)


def _copied_snapshot_field_keys(snapshot: str, source_text: str) -> set[str]:
    fields = _parse_snapshot_fields(snapshot)
    body = _strip_snapshot_discourse_openers(
        normalize_chinese_role_terms(
            strip_analyst_decision_summary(strip_feedback_snapshot(source_text))
        )
    )
    normalized_body = _normalize_overlap_text(body)
    copied_fields = set()
    if not normalized_body:
        return copied_fields

    for field_key, value in fields.items():
        cleaned_value = _strip_snapshot_discourse_openers(value)
        normalized_value = _normalize_overlap_text(cleaned_value)
        if len(normalized_value) < 30:
            continue
        if normalized_value in normalized_body:
            copied_fields.add(field_key)
            continue
        if len(normalized_value) >= 45 and normalized_value[:45] in normalized_body:
            copied_fields.add(field_key)
    return copied_fields


def _snapshot_fields_substantially_overlap(left: str, right: str) -> bool:
    normalized_left = _normalize_overlap_text(_strip_snapshot_discourse_openers(left))
    normalized_right = _normalize_overlap_text(_strip_snapshot_discourse_openers(right))
    if len(normalized_left) < 24 or len(normalized_right) < 24:
        return False
    if normalized_left == normalized_right:
        return True

    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if len(shorter) >= 30 and shorter in longer:
        return True

    similarity = SequenceMatcher(
        None,
        normalized_left[:800],
        normalized_right[:800],
    ).ratio()
    if similarity >= 0.72:
        return True

    left_topics = set(_extract_snapshot_topics(left))
    right_topics = set(_extract_snapshot_topics(right))
    shared_topics = left_topics & right_topics

    number_pattern = r"\d+(?:\.\d+)?(?:%|％|亿|万|元|天)?"
    left_numbers = set(re.findall(number_pattern, left))
    right_numbers = set(re.findall(number_pattern, right))
    shared_numbers = left_numbers & right_numbers

    if shared_topics and shared_numbers:
        return True

    prefix_len = min(len(shorter), len(longer), 60)
    return prefix_len >= 36 and shorter[:prefix_len] == longer[:prefix_len]


def _snapshot_values_are_near_duplicates(left: str, right: str) -> bool:
    normalized_left = _normalize_overlap_text(_strip_snapshot_discourse_openers(left))
    normalized_right = _normalize_overlap_text(_strip_snapshot_discourse_openers(right))
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    shorter, longer = sorted((normalized_left, normalized_right), key=len)
    if len(shorter) >= 24 and shorter in longer and len(shorter) >= int(len(longer) * 0.7):
        return True

    similarity = SequenceMatcher(
        None,
        normalized_left[:800],
        normalized_right[:800],
    ).ratio()
    return similarity >= 0.88


def _merge_snapshot_field_values(existing_value: str, new_value: str, chinese_snapshot: bool) -> str:
    existing_value = existing_value.strip()
    new_value = new_value.strip()
    if not existing_value:
        return new_value
    if not new_value:
        return existing_value
    if _snapshot_values_are_near_duplicates(existing_value, new_value):
        return existing_value if len(existing_value) >= len(new_value) else new_value
    separator = "；" if chinese_snapshot else "; "
    return f"{existing_value}{separator}{new_value}"


def _looks_like_snapshot_rebuttal(value: str) -> bool:
    normalized = normalize_chinese_role_terms((value or "").strip())
    if not normalized:
        return False
    if len(normalized) <= 80:
        return True

    keywords = (
        "对手", "空头分析师", "多头分析师", "激进分析师", "保守分析师", "中性分析师",
        "bull analyst", "bear analyst", "aggressive analyst", "conservative analyst",
        "neutral analyst", "忽略", "误判", "高估", "低估", "但", "然而", "却", "反驳",
        "质疑", "难以成立", "does not hold", "missed", "ignored", "however", "but ",
        "challenge", "rebut",
    )
    lowered = normalized.lower()
    return any(keyword in normalized or keyword in lowered for keyword in keywords)


def _looks_like_snapshot_new_content(value: str) -> bool:
    normalized = normalize_chinese_role_terms((value or "").strip())
    if not normalized:
        return False
    if len(normalized) <= 80:
        return True

    lowered = normalized.lower()
    bad_starts = (
        "空头分析师，", "多头分析师，", "激进分析师，", "保守分析师，", "中性分析师，",
        "bear analyst,", "bull analyst,", "aggressive analyst,", "conservative analyst,",
        "neutral analyst,",
        "我理解你对", "让我用", "各位", "大家", "我们必须",
    )
    if any(normalized.startswith(prefix) or lowered.startswith(prefix) for prefix in bad_starts):
        return False

    discourse_patterns = (
        r"^.{0,12}(?:分析师|manager)[，,:：]",
        r"^(?:我理解你对|让我用|各位|大家|首先|总的来说)",
    )
    return not any(re.search(pattern, normalized, re.IGNORECASE) for pattern in discourse_patterns)


def _looks_like_snapshot_verification(value: str) -> bool:
    normalized = normalize_chinese_role_terms((value or "").strip())
    if not normalized:
        return False

    if (
        normalized.startswith("**")
        or normalized.startswith("#")
        or "理性框架" in normalized
        or "decision framework" in normalized.lower()
    ):
        return False
    if len(normalized) <= 80:
        return True

    lowered = normalized.lower()
    keywords = (
        "跟踪", "验证", "观察", "等待", "确认", "关注", "若", "如果", "一旦", "突破", "跌破",
        "守稳", "高于", "低于", "阈值", "财报", "订单", "毛利率", "利润率", "量能", "成交量",
        "业绩", "指引", "track", "verify", "watch", "monitor", "confirm", "earnings",
        "orders", "margin", "volume", "guidance", "above", "below", "break", "if ",
    )
    if any(keyword in normalized or keyword in lowered for keyword in keywords):
        return True

    return False


def _snapshot_replacement_fields(snapshot: str, source_text: str) -> set[str]:
    fields = _parse_snapshot_fields(snapshot)
    replacement_fields = set(_copied_snapshot_field_keys(snapshot, source_text))

    new_and_rebuttal = fields.get("new_and_rebuttal", "")
    to_verify = fields.get("to_verify", "")

    if _snapshot_fields_substantially_overlap(new_and_rebuttal, to_verify):
        replacement_fields.add("to_verify")

    if new_and_rebuttal and not _looks_like_snapshot_new_content(new_and_rebuttal):
        replacement_fields.add("new_and_rebuttal")
    if to_verify and not _looks_like_snapshot_verification(to_verify):
        replacement_fields.add("to_verify")

    return replacement_fields


def is_feedback_snapshot_inferred(text: str) -> bool:
    """Return True when the displayed snapshot will be inferred from the body."""
    if not text or not text.strip():
        return True

    for marker in SNAPSHOT_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            snapshot = text[idx:].strip()
            return (
                _contains_placeholder_snapshot(snapshot)
                or _snapshot_has_missing_fields(snapshot)
                or bool(_snapshot_replacement_fields(snapshot, text))
            )
    return True


def _extract_snapshot_topics(body: str, chinese: bool | None = None) -> list[str]:
    topics = []
    use_chinese = _snapshot_output_is_chinese(chinese)
    patterns = (
        (r"库存|存货", "库存与备货压力" if use_chinese else "inventory positioning"),
        (r"需求|订单|出货", "需求与订单兑现" if use_chinese else "demand and order conversion"),
        (r"金价|黄金", "金价走势" if use_chinese else "gold price trend"),
        (r"并购|收购|整合", "并购进度" if use_chinese else "M&A execution"),
        (r"地缘|关税|出口|海外", "地缘与海外政策风险" if use_chinese else "geopolitical and export-policy risk"),
        (r"价格战|定价权", "价格战下的定价权" if use_chinese else "pricing power through the price war"),
        (r"财报|业绩|一季报|q[1-4]", "业绩验证" if use_chinese else "earnings confirmation"),
        (r"储能", "储能需求兑现" if use_chinese else "storage-demand conversion"),
        (r"1\.6t|良率", "1.6T良率爬坡" if use_chinese else "1.6T yield ramp"),
        (r"估值|pe|市盈率", "高估值消化能力" if use_chinese else "valuation absorption"),
        (r"毛利率|利润率", "毛利率变化" if use_chinese else "margin trend"),
        (r"支撑|底部|macd|金叉|死叉", "技术支撑与动量修复" if use_chinese else "technical support and momentum repair"),
        (r"成交量|量能|放量|缩量|vwma", "量能确认" if use_chinese else "volume confirmation"),
        (r"情绪|热度|社交媒体|抢筹", "情绪热度" if use_chinese else "sentiment heat"),
        (r"资本开支|capex", "资本开支验证" if use_chinese else "capex follow-through"),
        (r"波动|atr|止损", "波动率与止损边界" if use_chinese else "volatility and stop-loss bands"),
    )
    lowered = body.lower()
    for pattern, topic in patterns:
        if re.search(pattern, lowered, re.IGNORECASE):
            topics.append(topic)
    return topics


def _infer_feedback_snapshot_from_body(
    text: str,
    paraphrase: bool = False,
    chinese: bool | None = None,
) -> str:
    body = _strip_snapshot_discourse_openers(
        normalize_chinese_role_terms(
            strip_analyst_decision_summary(strip_feedback_snapshot(text))
        )
    )
    sentences = _extract_sentences(body)
    first = _condense_excerpt(sentences[0], 120) if sentences else _condense_excerpt(body, 120)
    second = _condense_excerpt(sentences[1], 120) if len(sentences) > 1 else first
    use_chinese = _snapshot_output_is_chinese(chinese)
    topics = _extract_snapshot_topics(body, chinese=use_chinese)

    if use_chinese:
        rating = _detect_risk_stance(text) or _detect_chinese_rating(text)
        if paraphrase:
            joined_topics = "、".join(topics[:3]) if topics else f"“{rating}”逻辑"
            follow_up_topics = "、".join(topics[:3]) if topics else "关键数据与风险触发条件"
            new_and_rebuttal = (
                f"本轮新增了对{joined_topics}的归因和执行含义说明，并据此反驳对手把这些信号"
                f"线性等同于单边风险的推断，说明其结论仍需结合兑现节奏、盈利韧性和价格信号综合判断。"
            )
            to_verify = f"下一轮继续跟踪{follow_up_topics}这几项指标是否同步验证；若关键数据没有改善，就需要重新评估“{rating}”判断。"
        else:
            new_and_rebuttal = (
                _pick_sentence(sentences, ("新增", "补充", "转向", "强调", "原因", "估值", "库存", "需求", "订单"))
                or first
                or f"本轮围绕“{rating}”补充了关键证据、风险边界和执行依据。"
            )
            to_verify = (
                _pick_sentence(sentences, ("跟踪", "验证", "观察", "等待", "确认", "关注"))
                or (second if second and second != new_and_rebuttal else "")
                or f"下一轮验证支持“{rating}”的关键数据和风险触发条件。"
            )
        return (
            "反馈快照:\n"
            f"- 立场: {rating}\n"
            f"- 本轮新增与反驳: {new_and_rebuttal}\n"
            f"- 待验证: {to_verify}"
        )

    rating = _detect_english_rating(text)
    if paraphrase:
        joined_topics = ", ".join(topics[:3]) if topics else f"the {rating} thesis"
        new_and_rebuttal = (
            f"This round added clearer causality and execution context around {joined_topics}, and used that"
            f" incremental evidence to challenge the opposing side's one-way risk interpretation."
        )
        to_verify = f"Next round should verify whether {joined_topics} continue to improve enough to sustain the {rating} stance."
    else:
        new_and_rebuttal = (
            _pick_sentence(sentences, ("added", "new", "shifted", "because", "valuation", "demand", "inventory", "orders"))
            or first
            or f"Added key evidence, risk boundaries, and execution rationale behind the {rating} call."
        )
        to_verify = (
            _pick_sentence(sentences, ("track", "verify", "watch", "wait", "confirm", "monitor"))
            or (second if second and second != new_and_rebuttal else "")
            or f"Verify core data assumptions, risk triggers, and timing conditions behind the {rating} stance."
        )
    return (
        "FEEDBACK SNAPSHOT:\n"
        f"- Stance: {rating}\n"
        f"- New this round & rebuttal: {new_and_rebuttal}\n"
        f"- To verify: {to_verify}"
    )

def extract_feedback_snapshot(text: str) -> str:
    """Extract the structured feedback snapshot block from an agent response."""
    if not text:
        if _is_chinese_output():
            return "反馈快照:\n- 立场: 暂无。\n- 本轮新增与反驳: 暂无。\n- 待验证: 暂无。"
        return "FEEDBACK SNAPSHOT:\n- Stance: None yet.\n- New this round & rebuttal: None yet.\n- To verify: None yet."

    for marker in SNAPSHOT_MARKERS:
        idx = text.rfind(marker)
        if idx != -1:
            snapshot = text[idx:].strip()
            source_is_chinese = _snapshot_uses_chinese(snapshot)
            if _contains_placeholder_snapshot(snapshot):
                return _infer_feedback_snapshot_from_body(
                    text,
                    paraphrase=True,
                    chinese=source_is_chinese,
                )
            normalized_snapshot = normalize_chinese_role_terms(snapshot)
            replacement_fields = _snapshot_replacement_fields(normalized_snapshot, text)
            has_missing_fields = _snapshot_has_missing_fields(normalized_snapshot)
            if has_missing_fields or replacement_fields:
                inferred_snapshot = _infer_feedback_snapshot_from_body(
                    text,
                    paraphrase=True,
                    chinese=source_is_chinese,
                )
                return _merge_snapshot_with_inferred(
                    normalized_snapshot,
                    inferred_snapshot,
                    replacement_fields=replacement_fields,
                    chinese=source_is_chinese,
                )
            return _format_snapshot_from_fields(
                _parse_snapshot_fields(normalized_snapshot),
                chinese=source_is_chinese,
            )

    return _infer_feedback_snapshot_from_body(text, paraphrase=True)


def save_snapshot_file(
    snapshot: str,
    role: str,
    ticker: str,
    trade_date: str,
    round_num: int,
) -> str:
    """Save the full snapshot text to a dedicated file and return its path."""
    from tradingagents.dataflows.config import get_config

    results_dir = get_config().get("results_dir", "./results")
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    snap_dir = os.path.join(results_dir, safe_ticker, trade_date, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    safe_role = role.lower().replace(" ", "_")
    filename = f"{safe_role}_round_{round_num}.md"
    path = os.path.join(snap_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {role} — Round {round_num} Snapshot\n\n")
        f.write(snapshot)
    return path


def load_snapshot_file(file_path: str) -> str:
    """Load snapshot content from a file. Returns empty string if missing."""
    if not file_path:
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return ""


def _clean_snapshot_display_value(value: str) -> str:
    if not value:
        return ""

    if "；" in value:
        separator = "；"
        segments = value.split("；")
    elif ";" in value:
        separator = "; "
        segments = value.split(";")
    else:
        separator = ""
        segments = [value]

    cleaned_segments = []
    for segment in segments:
        cleaned = _strip_snapshot_discourse_openers(segment.strip())
        if cleaned:
            cleaned_segments.append(cleaned)

    if not cleaned_segments:
        return ""
    if not separator:
        return cleaned_segments[0]
    return separator.join(cleaned_segments)


def make_display_snapshot(
    full_snapshot: str,
    file_path: str = "",
    include_link: bool = False,
) -> str:
    """Create a brief inline display of the snapshot.

    Snapshot files remain available through the stored ``*_snapshot_path`` state,
    but user-facing displays hide file links by default to keep the report clean.
    """
    fields = _parse_snapshot_fields(full_snapshot)
    labels = _snapshot_field_labels()
    field_keys = list(_snapshot_field_aliases().keys())

    lines = []
    for key, label in zip(field_keys, labels):
        value = fields.get(key, "").strip()
        if value:
            cleaned_value = _clean_snapshot_display_value(value)
            lines.append(f"- {label}: {cleaned_value}")

    display = "\n".join(lines) if lines else full_snapshot[:200]
    if include_link and file_path:
        link_label = "完整内容见" if _is_chinese_output() else "Full snapshot"
        display += f"\n({link_label}: {file_path})"
    return display


def build_history_turn(raw_content: str, role: str) -> str:
    """Preserve the visible turn content, including structured blocks, for round history."""
    cleaned = strip_role_prefix((raw_content or "").strip(), role)
    speaker = localize_role_name(role)
    if not cleaned:
        return f"{speaker}:"
    return f"{speaker}: {cleaned}"


def strip_role_prefix(text: str, role: str) -> str:
    """Remove self-labeling role prefixes that the LLM may inject.

    Handles four patterns:
    1. Leading markdown heading containing the role name, e.g. '# 多头分析师辩论：...'
    2. Leading 'role: ' / 'role：' prefix at the start of the body
    3. Mid-text paragraphs that open with 'role：' / 'role: ' (inline self-labeling)
    4. Leading greeting sentences directed at another role, e.g. '你好，空头分析师。'
    """
    if not text:
        return text

    import re

    candidates = _ROLE_BOTH_NAMES.get(role, {role, localize_role_name(role)})

    # 1. Strip leading markdown heading line(s) that contain the role name
    lines = text.split("\n")
    while lines and lines[0].lstrip().startswith("#") and any(
        name in lines[0] for name in candidates
    ):
        lines.pop(0)
    text = "\n".join(lines).lstrip()

    # 2. Strip leading 'role: ' / 'role：' prefix
    for name in candidates:
        for sep in ("：", ": ", ":", "- "):
            prefix = name + sep
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        else:
            continue
        break

    # 3. Remove mid-text self-labeling: 'role：' or 'role: ' at the start of any line
    pattern = "|".join(re.escape(name) for name in candidates)
    text = re.sub(r"(?m)^(?:" + pattern + r")[：:] *", "", text)

    # 4. Strip leading greeting sentences (e.g. '你好，空头分析师。' / '空头分析师，你好。')
    # Match one or two greeting sentences at the very start (up to first period/exclamation).
    _GREETING_RE = re.compile(
        r"^(?:"
        r"(?:你好|大家好|各位好|亲爱的)[，、,！!]?[^\n。！!?？]{0,30}[。！!?？\n]?"
        r"|[^\n，。！!?？]{0,20}[，,][你妳]好[。！!?？\n]?"
        r")\s*",
        re.UNICODE,
    )
    text = _GREETING_RE.sub("", text).lstrip()

    return text.strip()


def strip_feedback_snapshot(text: str) -> str:
    """Remove the feedback snapshot block from the visible argument body."""
    if not text:
        return ""
    best_idx = -1
    for marker in SNAPSHOT_MARKERS:
        idx = text.rfind(marker)
        if idx > best_idx:
            best_idx = idx
    if best_idx == -1:
        return text.strip()
    return text[:best_idx].strip()


def build_debate_brief(snapshots: dict[str, str], latest_speaker: str = "") -> str:
    """Build a compact cross-agent brief from the latest structured snapshots."""
    sections = []
    if latest_speaker:
        if _is_chinese_output():
            sections.append(f"最新更新来自: {localize_role_name(latest_speaker)}")
        else:
            sections.append(f"Latest update came from: {latest_speaker}")

    for speaker, snapshot in snapshots.items():
        if snapshot:
            if _is_chinese_output():
                sections.append(f"{localize_role_name(speaker)} 最新快照:\n{snapshot}")
            else:
                sections.append(f"{speaker} latest snapshot:\n{snapshot}")

    return "\n\n".join(sections).strip()


def synthesize_side_report(llm, role: str, full_history: str, snapshot: str) -> str:
    """Use the LLM to distill a debater's full multi-round history into a structured
    comprehensive position report for the manager to read.

    This replaces the raw 'last round only' approach with a coherent synthesis that
    captures argument evolution, strongest evidence, and final stance across all rounds.
    """
    if not full_history:
        return snapshot or ""

    role_label = localize_role_name(role) if _is_chinese_output() else role
    history_input = truncate_for_prompt(full_history, default_limit=12000)

    if _is_chinese_output():
        prompt = (
            f"以下是【{role_label}】在多轮辩论中的完整发言记录：\n\n"
            f"{history_input}\n\n"
            f"最新快照摘要（供参考）：\n{snapshot}\n\n"
            "请基于以上全部内容，生成一份结构化的【综合立场报告】，涵盖：\n"
            "1. **最终立场**：明确的评级及核心判断（1-2句）\n"
            "2. **核心论点**：跨轮次最有力的3-5个论点，附关键数据或事实支撑\n"
            "3. **主要反驳**：对对手观点最有效的反驳（2-3条）\n"
            "4. **立场演变**：若立场在辩论过程中有调整，简述原因\n"
            "5. **核心风险/机会**：一句话点明最关键的尾部风险或上行机会\n"
            "报告应客观、具体、有数据支撑，避免重复，控制在500字以内。"
        )
    else:
        prompt = (
            f"Below is the complete multi-round debate record from [{role_label}]:\n\n"
            f"{history_input}\n\n"
            f"Latest snapshot (for reference):\n{snapshot}\n\n"
            "Based on the full record above, produce a structured **Comprehensive Position Report** covering:\n"
            "1. **Final Stance**: Clear rating and core thesis (1-2 sentences)\n"
            "2. **Key Arguments**: The 3-5 strongest arguments across all rounds, with data/evidence\n"
            "3. **Main Rebuttals**: Most effective counter-arguments to the opponent (2-3 points)\n"
            "4. **Stance Evolution**: If the position shifted during debate, briefly explain why\n"
            "5. **Key Risk/Opportunity**: One sentence on the most critical tail risk or upside\n"
            "Be concrete, data-anchored, non-repetitive. Max 400 words."
        )

    try:
        response = llm.invoke(prompt)
        return extract_text_content(response.content)
    except Exception:
        # Fall back to truncated raw history if synthesis fails
        return truncate_for_prompt(full_history, default_limit=4000)


def _resolve_company_name(ticker: str) -> str:
    """Best-effort lookup of the company name for a ticker.

    Tries Tushare first (accurate for Chinese A-shares / HK stocks), then
    falls back to yfinance.  Returns an empty string when both fail so the
    caller can decide whether to include the name.
    """
    # --- Tushare (A-share / HK) ---
    suffix = ticker.rsplit(".", 1)[-1].upper() if "." in ticker else ""
    if suffix in ("SH", "SZ", "BJ", "HK"):
        try:
            import tushare as ts
            import os
            token = os.getenv("TUSHARE_TOKEN", "")
            if token:
                ts.set_token(token)
                pro = ts.pro_api()
                # Normalize to tushare canonical format (601899.SH / 600000.SH)
                ts_code = ticker.upper()
                df = pro.stock_basic(ts_code=ts_code, fields="ts_code,name")
                if not df.empty:
                    return df["name"].iloc[0]
        except Exception:
            pass

    # --- yfinance fallback ---
    try:
        import yfinance as yf
        _YF_MAP = {"SH": "SS", "SSE": "SS"}
        if "." in ticker:
            code, sfx = ticker.rsplit(".", 1)
            yf_sym = f"{code}.{_YF_MAP.get(sfx.upper(), sfx)}".upper()
        else:
            yf_sym = ticker.upper()
        info = yf.Ticker(yf_sym).info
        name = info.get("longName") or info.get("shortName") or ""
        return name
    except Exception:
        pass

    return ""


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    name = _resolve_company_name(ticker)
    name_clause = f" ({name})" if name else ""
    return (
        f"The instrument to analyze is `{ticker}`{name_clause}. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
CHINESE_ROLE_TERM_REPLACEMENTS = {
    "熊派分析师": "空头分析师",
    "熊派投资者": "空头投资者",
    "熊观点": "空头观点",
    "熊派": "空头",
    "牛派分析师": "多头分析师",
    "牛派投资者": "多头投资者",
    "牛观点": "多头观点",
    "牛派": "多头",
}
CHINESE_ROLE_TERM_PATTERN = re.compile(
    "|".join(
        sorted(
            (re.escape(term) for term in CHINESE_ROLE_TERM_REPLACEMENTS),
            key=len,
            reverse=True,
        )
    )
)
