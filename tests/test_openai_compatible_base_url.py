import unittest
from unittest.mock import patch

from tradingagents.llm_clients.factory import create_llm_client
from tradingagents.llm_clients.openai_client import OpenAIClient


class OpenAICompatibleBaseUrlTests(unittest.TestCase):
    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_ollama_provider_respects_explicit_base_url(self, mock_chat):
        client = OpenAIClient(
            "qwen3:latest",
            base_url="http://localhost:4000/v1",
            provider="ollama",
        )
        client.get_llm()

        kwargs = mock_chat.call_args[1]
        self.assertEqual(kwargs["base_url"], "http://localhost:4000/v1")
        self.assertEqual(kwargs["api_key"], "ollama")

    @patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI")
    def test_factory_preserves_ollama_alias_model_name(self, mock_chat):
        client = create_llm_client(
            provider="ollama",
            model="Qwen3.5-35B-A3B",
            base_url="http://localhost:4000/v1",
        )
        client.get_llm()

        kwargs = mock_chat.call_args[1]
        self.assertEqual(kwargs["model"], "Qwen3.5-35B-A3B")


if __name__ == "__main__":
    unittest.main()
