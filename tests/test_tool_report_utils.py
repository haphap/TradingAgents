import unittest

from tradingagents.tool_report_utils import run_tool_report_chain


class _FakeResponse:
    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeBoundLLM:
    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, _messages):
        return self._responses.pop(0)


class _FakeLLM:
    def __init__(self, tool_responses, fallback_responses=None):
        self._tool_bound = _FakeBoundLLM(tool_responses)
        self._fallback = _FakeBoundLLM(fallback_responses or [])

    def bind_tools(self, _tools):
        return self._tool_bound

    def invoke(self, _messages):
        return self._fallback.invoke(_messages)


class _FakePrompt:
    def partial(self, **_kwargs):
        return self

    def __or__(self, runnable):
        return runnable


class ToolReportUtilsTests(unittest.TestCase):
    def test_returns_tool_response_when_more_tools_needed(self):
        prompt = _FakePrompt()
        llm = _FakeLLM([_FakeResponse(tool_calls=[{"name": "get_stock_data"}])])

        result, report = run_tool_report_chain(
            prompt,
            llm,
            tools=["tool"],
            messages=["state"],
            system_message="sys",
        )

        self.assertEqual(report, "")
        self.assertEqual(len(result.tool_calls), 1)

    def test_retries_empty_final_response_without_tools(self):
        prompt = _FakePrompt()
        llm = _FakeLLM(
            [_FakeResponse(content="")],
            [_FakeResponse(content=[{"type": "text", "text": "Final report"}])],
        )

        result, report = run_tool_report_chain(
            prompt,
            llm,
            tools=["tool"],
            messages=["state"],
            system_message="sys",
        )

        self.assertEqual(report, "Final report")
        self.assertEqual(result.content[0]["text"], "Final report")


if __name__ == "__main__":
    unittest.main()
