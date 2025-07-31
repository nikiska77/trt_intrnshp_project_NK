"""
Agentic sampling loop that calls the Anthropic API and local implementation of anthropic-defined computer use tools.
"""

import platform
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum
from typing import Any, cast, Callable, List
import openai
from openai import OpenAI
import os
import requests
import base64
import json
import re
# from computer_use_demo.nebius_config import nebius_config
import httpx
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicVertex,
    APIError,
    APIResponseValidationError,
    APIStatusError,
)
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
    BetaContentBlockParam,
    BetaImageBlockParam,
    BetaMessage,
    BetaMessageParam,
    BetaTextBlock,
    BetaTextBlockParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
)

from .tools import (
    TOOL_GROUPS_BY_VERSION,
    ToolCollection,
    ToolResult,
    ToolVersion,
)
from tools.bashnebius import NebiusBashTool

PROMPT_CACHING_BETA_FLAG = "prompt-caching-2024-07-31"


class APIProvider(StrEnum):
    ANTHROPIC = "anthropic"
    BEDROCK = "bedrock"
    VERTEX = "vertex"
    NEBIUS = "nebius"


# This system prompt is optimized for the Docker environment in this repository and
# specific tool combinations enabled.
# We encourage modifying this system prompt to ensure the model has context for the
# environment it is running in, and to provide any additional information that may be
# helpful for the task at hand.
SYSTEM_PROMPT = f"""<SYSTEM_CAPABILITY>
* You are utilising an Ubuntu virtual machine using {platform.machine()} architecture with internet access. You use tools by writing bash code inside ```bash blocks. Use bash syntax only for tool invocation.
* You can feel free to install Ubuntu applications with your bash tool. Use curl instead of wget.
* To open firefox, please just click on the firefox icon.  Note, firefox-esr is what is installed on your system.
* Using bash tool you can start GUI applications, but you need to set export DISPLAY=:1 and use a subshell. For example "(DISPLAY=:1 xterm &)". GUI apps run with bash tool will appear within your desktop environment, but they may take some time to appear. Take a screenshot to confirm it did.
* When using your bash tool with commands that are expected to output very large quantities of text, redirect into a tmp file and use str_replace_based_edit_tool or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output.
* When viewing a page it can be helpful to zoom out so that you can see everything on the page.  Either that, or make sure you scroll down to see everything before deciding something isn't available.
* When using your computer function calls, they take a while to run and send back to you.  Where possible/feasible, try to chain multiple of these calls all into one function calls request.
* The current date is {datetime.today().strftime('%A, %B %-d, %Y')}.
</SYSTEM_CAPABILITY>

<IMPORTANT>
* When using Firefox, if a startup wizard appears, IGNORE IT.  Do not even click "skip this step".  Instead, click on the address bar where it says "Search or enter address", and enter the appropriate search term or URL there.
* If the item you are looking at is a pdf, if after taking a single screenshot of the pdf it seems that you want to read the entire document instead of trying to continue to read the pdf from your screenshots + navigation, determine the URL, use curl to download the pdf, install and use pdftotext to convert it to a text file, and then read that text file directly with your str_replace_based_edit_tool.
</IMPORTANT>"""


# Client initialization for  Nebius AI
def init_nebius_client():
    # load_dotenv()
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("Missing Nebius API credentials. Check your .env file")
    # print(api_key)
    return OpenAI(
        api_key=api_key,
        # base_url=os.getenv("BASE_URL"),
        base_url="https://api.studio.nebius.com/v1/",
    )
def test_nebius_connection():
    client = init_nebius_client()
    try:
        response = client.models.list()
        print("Connection successful. Available models:", [m.id for m in response.data])
    except Exception as e:
        print("Connection failed:", str(e))


def convert_messages(system_prompt: str, messages: List[BetaMessageParam]):
    result = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, list):
            texts = [block["text"] for block in content if block["type"] == "text"]
            text = "\n".join(texts)
        else:
            text = content
        result.append({"role": role, "content": text})
    return result


def convert_tool_use_blocks(tool_uses: list[dict[str, Any]]) -> list[BetaToolResultBlockParam]:
    return [
        BetaToolResultBlockParam(
            tool_use_id=tool_use["id"],
            type="tool_result",
            content=tool_use["result"].output if tool_use["result"].output else tool_use["result"].error,
            is_error=bool(tool_use["result"].error),
        )
        for tool_use in tool_uses
    ]


def convert_tool_call_to_result(tool_call):
    return tool_call.function.name, json.loads(tool_call.function.arguments)


async def sampling_loop(
    *,
    model: str,
    provider: APIProvider,
    system_prompt_suffix: str,
    messages: list[BetaMessageParam],
    output_callback: Callable[[BetaContentBlockParam], None],
    tool_output_callback: Callable[[ToolResult, str], None],
    api_response_callback: Callable[
        [httpx.Request, httpx.Response | object | None, Exception | None], None
    ],
    api_key: str,
    only_n_most_recent_images: int | None = None,
    max_tokens: int = 4096,
    tool_version: ToolVersion,
    thinking_budget: int | None = None,
    token_efficient_tools_beta: bool = False,
):
    """
    Agentic sampling loop for the assistant/tool interaction of computer use.
    """
    tool_group = TOOL_GROUPS_BY_VERSION[tool_version]
    tool_collection = ToolCollection(*(ToolCls() for ToolCls in tool_group.tools))
    print("Active tools:", list(tool_collection.tool_map.keys()))
    tools_nebius = [
        # NebiusComputerTool(),
        # NebiusEditTool(),
        NebiusBashTool()
    ]
    system = BetaTextBlockParam(
        type="text",
        text=f"{SYSTEM_PROMPT}{' ' + system_prompt_suffix if system_prompt_suffix else ''}",
    )

    while True:
        enable_prompt_caching = False
        betas = [tool_group.beta_flag] if tool_group.beta_flag else []
        if token_efficient_tools_beta:
            betas.append("token-efficient-tools-2025-02-19")
        image_truncation_threshold = only_n_most_recent_images or 0
        extra_body = {}
        if thinking_budget:
            # Ensure we only send the required fields for thinking
            extra_body = {
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}
            }

        if provider == APIProvider.NEBIUS:  # I added this code
            model = "Qwen/Qwen2.5-VL-72B-Instruct"
            # model= "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
            converted_messages = convert_messages(system["text"], messages) # Messages conversion to OpenAI format
            # openai_tools = convert_tools_for_openai(tool_collection)
            # print(tool_collection)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            body = {
                "model": model,
                "messages": converted_messages,
                "max_tokens": max_tokens,
                "temperature": 0.3,
                "tools": [tool.to_params() for tool in tools_nebius],
                "tool_choice": "auto",
                "stream": False,
            }

            try:
                async with httpx.AsyncClient() as client:
                    r = await client.post("https://api.studio.nebius.com/v1/chat/completions", headers=headers,
                                          json=body)
                    api_response_callback(r.request, r, None)
                    print("Nebius raw response:", r.text)
                    data = r.json()

                    choice = data["choices"][0]
                    assistant_content = choice.get("message", {}).get("content")
                    if not assistant_content:
                        return messages
                    block = BetaTextBlockParam(type="text", text=assistant_content)
                    output_callback(block)
                    messages.append({"role": "assistant", "content": [block]})
                    print(tool_collection.tools)
                    # try extract tool_use manually from generated response if exists
                    # check if there's a bash command to simulate tool call
                    # match = re.search(r"```bash\\n(.+?)\\n```", assistant_content, re.DOTALL)
                    match = re.search(r"```bash\s*\n(.*?)\n```", assistant_content, re.DOTALL)

                    if match:
                        command = match.group(1).strip()
                        print("command", command)
                        bash_tool = tool_collection.tool_map.get("bash")
                        if bash_tool:
                            result = await bash_tool(command=command)
                            tool_id = f"toolu_bash"
                            tool_result = BetaToolResultBlockParam(
                                tool_use_id=tool_id,
                                type="tool_result",
                                content=result.output if result.output else result.error,
                                is_error=bool(result.error),
                            )
                            tool_output_callback(result, tool_id)
                            messages.append({"role": "user", "content": [tool_result]})

                    return messages
            except Exception as e:
                api_response_callback(None, None, e)
                return messages

                # if response.choices and response.choices[0].message.content:
                #     return response.choices[0].message.content
                # return "Received empty response from Nebius API"
            #     for chunk in response:
            #         delta = chunk.choices[0].delta
            #         if "content" in delta:
            #             full_response_text += delta["content"]
            #             block = BetaTextBlockParam(type="text", text=delta["content"])
            #             output_callback(block)
            #     messages.append({
            #         "role": "assistant",
            #         "content": [BetaTextBlockParam(type="text", text=full_response_text)],
            #     })
            #     print(messages)
            #     return messages
            # except client.PermissionDeniedError as e:
            #     print(f"Permission denied: {e}")
            #     raise ValueError("Check your Nebius API key and permissions")
            #
            # except client.APIError as e:
            #     print(f"Nebius API error: {e}")
            #     raise


        # Call the API
        # we use raw_response to provide debug information to streamlit. Your
        # implementation may be able call the SDK directly with:
        # `response = client.messages.create(...)` instead.

        # print("Sending to Nebius:", json.dumps(openai_messages, indent=2))
        # print("Tools:", tool_collection.to_params())
        # test_nebius_connection()

            # return response.choices[0].message.content
        if provider == APIProvider.ANTHROPIC:
            client = Anthropic(api_key=api_key, max_retries=4)
            enable_prompt_caching = True
        elif provider == APIProvider.VERTEX:
            client = AnthropicVertex()
        elif provider == APIProvider.BEDROCK:
            client = AnthropicBedrock()

        if enable_prompt_caching:
            betas.append(PROMPT_CACHING_BETA_FLAG)
            _inject_prompt_caching(messages)
            # Because cached reads are 10% of the price, we don't think it's
            # ever sensible to break the cache by truncating images
            only_n_most_recent_images = 0
            # Use type ignore to bypass TypedDict check until SDK types are updated
            system["cache_control"] = {"type": "ephemeral"}  # type: ignore

        if only_n_most_recent_images:
            _maybe_filter_to_n_most_recent_images(
                messages,
                only_n_most_recent_images,
                min_removal_threshold=image_truncation_threshold,
            )

        try:
            raw_response = client.beta.messages.with_raw_response.create(
                    max_tokens=max_tokens,
                    messages=messages,
                    model=model,
                    system=[system],
                    tools=tool_collection.to_params(),
                    betas=betas,
                    extra_body=extra_body,
                )
        except (APIStatusError, APIResponseValidationError) as e:
            api_response_callback(e.request, e.response, e)
            return messages
        except APIError as e:
            api_response_callback(e.request, e.body, e)
            return messages

        api_response_callback(
            raw_response.http_response.request, raw_response.http_response, None
        )

        response = raw_response.parse()

        response_params = _response_to_params(response)
        messages.append(
            {
                "role": "assistant",
                "content": response_params,
            }
        )

        tool_result_content: list[BetaToolResultBlockParam] = []
        for content_block in response_params:
            output_callback(content_block)
            if content_block["type"] == "tool_use":
                result = await tool_collection.run(
                    name=content_block["name"],
                    tool_input=cast(dict[str, Any], content_block["input"]),
                )
                tool_result_content.append(
                    _make_api_tool_result(result, content_block["id"])
                )
                tool_output_callback(result, content_block["id"])

        if not tool_result_content:
            return messages

        messages.append({"content": tool_result_content, "role": "user"})


def _maybe_filter_to_n_most_recent_images(
    messages: list[BetaMessageParam],
    images_to_keep: int,
    min_removal_threshold: int,
):
    """
    With the assumption that images are screenshots that are of diminishing value as
    the conversation progresses, remove all but the final `images_to_keep` tool_result
    images in place, with a chunk of min_removal_threshold to reduce the amount we
    break the implicit prompt cache.
    """
    if images_to_keep is None:
        return messages

    tool_result_blocks = cast(
        list[BetaToolResultBlockParam],
        [
            item
            for message in messages
            for item in (
                message["content"] if isinstance(message["content"], list) else []
            )
            if isinstance(item, dict) and item.get("type") == "tool_result"
        ],
    )

    total_images = sum(
        1
        for tool_result in tool_result_blocks
        for content in tool_result.get("content", [])
        if isinstance(content, dict) and content.get("type") == "image"
    )

    images_to_remove = total_images - images_to_keep
    # for better cache behavior, we want to remove in chunks
    images_to_remove -= images_to_remove % min_removal_threshold

    for tool_result in tool_result_blocks:
        if isinstance(tool_result.get("content"), list):
            new_content = []
            for content in tool_result.get("content", []):
                if isinstance(content, dict) and content.get("type") == "image":
                    if images_to_remove > 0:
                        images_to_remove -= 1
                        continue
                new_content.append(content)
            tool_result["content"] = new_content


def _response_to_params(
    response: BetaMessage,
) -> list[BetaContentBlockParam]:
    res: list[BetaContentBlockParam] = []
    for block in response.content:
        if isinstance(block, BetaTextBlock):
            if block.text:
                res.append(BetaTextBlockParam(type="text", text=block.text))
            elif getattr(block, "type", None) == "thinking":
                # Handle thinking blocks - include signature field
                thinking_block = {
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", None),
                }
                if hasattr(block, "signature"):
                    thinking_block["signature"] = getattr(block, "signature", None)
                res.append(cast(BetaContentBlockParam, thinking_block))
        else:
            # Handle tool use blocks normally
            res.append(cast(BetaToolUseBlockParam, block.model_dump()))
    return res


def _inject_prompt_caching(
    messages: list[BetaMessageParam],
):
    """
    Set cache breakpoints for the 3 most recent turns
    one cache breakpoint is left for tools/system prompt, to be shared across sessions
    """

    breakpoints_remaining = 3
    for message in reversed(messages):
        if message["role"] == "user" and isinstance(
            content := message["content"], list
        ):
            if breakpoints_remaining:
                breakpoints_remaining -= 1
                # Use type ignore to bypass TypedDict check until SDK types are updated
                content[-1]["cache_control"] = BetaCacheControlEphemeralParam(  # type: ignore
                    {"type": "ephemeral"}
                )
            else:
                content[-1].pop("cache_control", None)
                # we'll only every have one extra turn per loop
                break


def _make_api_tool_result(
    result: ToolResult, tool_use_id: str
) -> BetaToolResultBlockParam:
    """Convert an agent ToolResult to an API ToolResultBlockParam."""
    tool_result_content: list[BetaTextBlockParam | BetaImageBlockParam] | str = []
    is_error = False
    if result.error:
        is_error = True
        tool_result_content = _maybe_prepend_system_tool_result(result, result.error)
    else:
        if result.output:
            tool_result_content.append(
                {
                    "type": "text",
                    "text": _maybe_prepend_system_tool_result(result, result.output),
                }
            )
        if result.base64_image:
            tool_result_content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": result.base64_image,
                    },
                }
            )
    return {
        "type": "tool_result",
        "content": tool_result_content,
        "tool_use_id": tool_use_id,
        "is_error": is_error,
    }


def _maybe_prepend_system_tool_result(result: ToolResult, result_text: str):
    if result.system:
        result_text = f"<system>{result.system}</system>\n{result_text}"
    return result_text
