from tradingagents.content_utils import extract_text_content


_FINAL_REPORT_FALLBACK = (
    " You have already gathered all required data. "
    "Do not call any tools again. Write the final report now based only on the "
    "information already present in the conversation."
)


def run_tool_report_chain(prompt_template, llm, tools, messages, **prompt_kwargs):
    """Run a tool-enabled analyst chain and recover from empty final responses."""
    base_prompt = prompt_template.partial(**prompt_kwargs)
    result = (base_prompt | llm.bind_tools(tools)).invoke(messages)

    if getattr(result, "tool_calls", None):
        return result, ""

    report = extract_text_content(getattr(result, "content", None))
    if report:
        return result, report

    fallback_kwargs = dict(prompt_kwargs)
    fallback_kwargs["system_message"] = (
        f"{prompt_kwargs['system_message']}{_FINAL_REPORT_FALLBACK}"
    )
    fallback_prompt = prompt_template.partial(**fallback_kwargs)
    fallback_result = (fallback_prompt | llm).invoke(messages)
    fallback_report = extract_text_content(getattr(fallback_result, "content", None))

    if fallback_report:
        return fallback_result, fallback_report

    return result, ""
