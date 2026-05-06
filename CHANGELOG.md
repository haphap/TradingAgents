# Changelog

## [Unreleased]

### Added
- **Industry Research Analyst** (`broker_research_analyst`): Cross-analyses institutional industry research reports with focus on consensus views, divergences, blind spots, policy impact, and supply-chain dynamics
- **Stock Research Analyst** (`stock_research_analyst`): Cross-analyses individual stock research reports with focus on ratings, target prices, earnings estimates, valuation methods, and catalysts
- `get_stock_research` tool and tushare `get_stock_reports()` function for fetching individual stock research reports (`report_type="个股研报"`)
- Industry report lookup now uses stock's industry name (`ind_name`) instead of ticker (`ts_code`) via `stock_basic` API
- Research reports (`research_report`, `stock_report`) injected into Research Manager, Trader, and Portfolio Manager prompts
- "Blind spots and missing questions" analysis dimension in both research analyst prompts
- Detailed output quality requirements: specific price levels, volume thresholds, earnings verification metrics, and catalyst confirmation conditions for Research Manager, Trader, and Portfolio Manager
- Wider 120-day fallback search when no reports found in requested date range, with helpful error messages

### Changed
- **Renamed**: "券商研报分析师" / "Broker Research Analyst" → "行业研究分析师" / "Industry Research Analyst" across all display names, prompts, and CLI labels
- Industry research reports now fetched by stock's industry name (e.g., "通信设备") instead of stock ticker
- Research analyst prompts emphasize cross-analysis over summarization: root cause of disagreements, what brokers missed, unanswered questions
- Portfolio Manager and Trader prompts require concrete thresholds (volume as multiple of 20-day average, specific gross margin/ROE/order growth levels, catalyst confirmation conditions) instead of vague phrases like "watch volume"

### Fixed
- "No industry research reports found" error for stocks whose industry has coverage — now correctly looks up industry name via `stock_basic` API
- Graph node name mismatch after renaming "Broker Research" to "Industry Research" in `conditional_logic.py`
