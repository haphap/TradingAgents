import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from tradingagents.content_utils import extract_text_content


SchemaT = TypeVar("SchemaT", bound=BaseModel)
logger = logging.getLogger(__name__)


def bind_structured(llm: Any, schema: type[SchemaT]) -> Any:
    if not hasattr(llm, "with_structured_output"):
        raise NotImplementedError("LLM does not support structured output binding.")
    return llm.with_structured_output(schema)


def invoke_structured_or_freetext(llm: Any, prompt: Any, schema: type[SchemaT]) -> SchemaT | str:
    try:
        structured_llm = bind_structured(llm, schema)
        result = structured_llm.invoke(prompt)
        if isinstance(result, schema):
            return result
        if isinstance(result, dict):
            return schema.model_validate(result)
        if isinstance(result, BaseModel):
            return schema.model_validate(result.model_dump())
    except (AttributeError, NotImplementedError, TypeError, ValueError) as exc:
        logger.debug("Structured output unavailable, falling back to free text: %s", exc)

    response = llm.invoke(prompt)
    return extract_text_content(getattr(response, "content", response))
