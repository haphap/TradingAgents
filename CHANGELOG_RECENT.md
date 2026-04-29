# Recent Changelog

## 0. `Unreleased` `Improve structured reporting and upstream sync`
Date: 2026-04-29

- Upstream-aligned runtime and parsing:
  - Added shared bilingual 5-tier rating parsing in `tradingagents/agents/utils/rating.py`.
  - Switched `tradingagents/graph/signal_processing.py` to deterministic shared rating parsing instead of LLM-dependent extraction.
  - Reused the shared parser in `tradingagents/agents/utils/memory.py` so memory-log ratings match Research Manager / Portfolio Manager / signal processor output.
  - Refactored `tradingagents/agents/utils/structured.py` and manager / trader wiring to use pre-bound structured output with explicit free-text fallback behavior.
  - Added `scripts/smoke_structured_output.py` for manual structured-output smoke testing.
- Report consistency and localization:
  - Hardened `tradingagents/agents/schemas.py` so Research Manager, Trader, and Portfolio Manager outputs keep a single consistent 5-tier recommendation across rendered sections, snapshot stance, and final proposal lines.
  - Replaced placeholder-like structured text with safe fallback prose when models emit schema descriptions instead of real analysis.
  - Standardized Chinese-visible finance wording, translated mixed English finance terms, and normalized visible numeric expressions to Arabic numerals where appropriate.
  - Reworked assembled-report heading numbering in `cli/main.py` to the official hierarchy style:
    - `一、`
    - `（一）`
    - `1.`
    - `(1)`
    - `①`
- Risk, manager, and trader output quality:
  - Normalized visible risk recommendations to canonical actions so displayed action text no longer conflicts with the canonical rating.
  - Strengthened risk-analyst prompts so free-form risk suggestions map cleanly to the 5-tier scale.
  - Expanded Research Manager and Portfolio Manager prompt and render constraints so `辩论结论` / `行为逻辑` become full analytical paragraphs instead of terse restatements.
  - Expanded Trader execution-plan and risk-control defaults so reports explain support definitions, volume thresholds, catalyst confirmation, and concrete trigger conditions instead of generic “watch support / watch volume” wording.
- Fundamentals-report correctness:
  - Hardened `tradingagents/agents/analysts/fundamentals_analyst.py` so final reports are rewritten when they falsely claim balance-sheet, guidance, forward-valuation, or peer-sample data is missing while the tool output already contains it.
  - Tightened rewrite instructions so historical guidance that is too old for forward-PE calculation is described accurately rather than being misreported as “no guidance”.
- Tests:
  - Added `tests/test_signal_processing.py`.
  - Expanded regression coverage in:
    - `tests/test_context_memory_optimization.py`
    - `tests/test_fundamentals_analyst.py`
    - `tests/test_output_language_propagation.py`
    - `tests/test_cli_round_formatting.py`
    - `tests/test_structured_agents.py`
    - `tests/test_memory_log.py`

Impact:
- Structured reports are more internally consistent, more detailed, and less likely to leak placeholder or contradictory model output.
- Fundamentals reports now better respect the actual tool data instead of inventing missing-data narratives.
- Rating vocabulary is unified across structured managers, signal extraction, and deferred memory logging.

## 1. `ba7cb2f` `Harden inferred rating detection`
Date: 2026-04-03 16:50 +0800

- Rating inference:
  - Hardened inferred rating detection in `tradingagents/agents/utils/agent_utils.py`.
  - Chinese rating detection now uses:
    - explicit structured patterns first
    - heuristic patterns second
    - negation filtering to avoid false positives like `不建议买入`
  - English rating detection now applies the same negation-aware logic to avoid false positives like `do not recommend buy`.
- Snapshot inference:
  - Improved fallback snapshot generation so inferred `当前观点` / `Current thesis` is driven by rating detection instead of naive substring matches.
- Tests:
  - Expanded `tests/test_context_memory_optimization.py` with negation regression coverage for both Chinese and English.

Impact:
- Reduces BUY/HOLD/SELL misclassification in inferred snapshots.
- Makes fallback snapshot language closer to actual trading intent.
- Lowers risk of negated recommendations being interpreted as positive calls.

## 2. `9d9a159` `Refine Chinese debate output and data routing`
Date: 2026-04-03 16:26 +0800

- CLI output and reporting:
  - Reworked debate display in `cli/main.py` to format Research Team and Risk Management Team output by round instead of dumping cumulative per-analyst history.
  - Added round-grouped rendering with localized Chinese titles:
    - `第 N 轮`
    - `研究经理结论`
    - `投资组合经理结论`
    - localized analyst/researcher headers
  - Added round review blocks:
    - `本轮复盘` for model-authored snapshots
    - `自动复盘` for inferred fallback snapshots
- Chinese output quality:
  - Tightened researcher and manager prompt instructions so Chinese-mode outputs use Chinese body text and Chinese final conclusion titles.
  - Standardized Chinese role wording and normalized undesirable variants such as `熊派分析师` to preferred forms like `空头分析师`.
  - Added localized snapshot labels, debate brief labels, rating terms, and final proposal handling.
- Snapshot handling:
  - Added structured feedback snapshot extraction, snapshot stripping, debate brief building, and fallback inference utilities in `tradingagents/agents/utils/agent_utils.py`.
  - Improved snapshot display so each round can show thesis changes, rationale, rebuttal, and next-round lesson.
- Debate routing correctness:
  - Fixed localized routing sensitivity in `tradingagents/graph/conditional_logic.py` by relying on internal speaker state instead of localized response prefixes.
- Data vendor routing:
  - Introduced explicit vendor routing improvements across dataflows:
    - added `tradingagents/dataflows/tushare.py`
    - added `tradingagents/dataflows/opencli_news.py`
    - added `tradingagents/dataflows/brave_news.py`
    - added `tradingagents/dataflows/exceptions.py`
  - Updated interface and provider selection logic to prefer:
    - `tushare` for A-share price/fundamentals
    - `opencli`-based news/social aggregation where configured
- LLM/client/config updates:
  - Updated model validation and provider handling in:
    - `tradingagents/llm_clients/model_catalog.py`
    - `tradingagents/llm_clients/openai_client.py`
    - `tradingagents/default_config.py`
  - Added support-related changes for local/compatible backends and base URL handling.
- Repo hygiene:
  - Added generated-output ignores in `.gitignore`:
    - `results/`
    - `reports/`
- Tests:
  - Added broad regression coverage:
    - `tests/test_cli_provider_selection.py`
    - `tests/test_cli_round_formatting.py`
    - `tests/test_conditional_logic_localization.py`
    - `tests/test_context_memory_optimization.py`
    - `tests/test_data_vendor_routing.py`
    - `tests/test_openai_compatible_base_url.py`
    - `tests/test_opencli_news.py`
    - `tests/test_output_language_propagation.py`
    - `tests/test_signal_processing_localization.py`

Impact:
- Chinese-mode output is substantially more readable and consistent.
- Debate history is now usable in the CLI and exported reports.
- Vendor routing is more deterministic for A-share and news/social workflows.
- Localization no longer breaks internal debate routing.

## Combined Summary

- `9d9a159` delivered the large functional pass: localization, CLI debate formatting, data routing, and test coverage.
- `ba7cb2f` hardened one of the remaining weak spots: inferred rating detection under negated Chinese/English wording.
