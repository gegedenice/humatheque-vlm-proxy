import os
import json
import time
import uuid
import base64
import binascii
import io
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

MODEL_RUNNER_URL = os.environ.get(
    "VLM_RUNNER_URL",
    "http://host.docker.internal:12434"
)

MODEL_NAME = os.environ.get(
    "VLM_NAME",
    "hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL"
)

MODEL_ALIAS = os.environ.get("VLM_ALIAS", "Qwen3-VL-8B-Instruct-GGUF")
MAX_IMAGE_SIDE = int(os.environ.get("VLM_MAX_IMAGE_SIDE", "1024"))

app = FastAPI(title="Humatheque VLM Proxy")
RESPONSES_STORE = {}
RESPONSES_CHAT_HISTORY = {}


def json_or_error_response(response, upstream_name="model_runner"):
    try:
        return response.json()
    except json.JSONDecodeError:
        message = response.text.strip() or response.reason_phrase
        return {
            "error": {
                "message": message,
                "type": f"{upstream_name}_error",
                "status_code": response.status_code
            }
        }


async def request_json(request: Request):
    try:
        return await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON request body: {exc}")


def backend_model_candidates(model_name: str) -> set[str]:
    candidates = {model_name}

    if model_name.startswith("hf.co/"):
        candidates.add(model_name.replace("hf.co/", "huggingface.co/", 1))
    elif model_name.startswith("huggingface.co/"):
        candidates.add(model_name.replace("huggingface.co/", "hf.co/", 1))

    return {candidate.lower() for candidate in candidates}


def model_for_runner(model_name: str) -> str:
    if MODEL_ALIAS and model_name == MODEL_ALIAS:
        return MODEL_NAME
    if model_name.lower() in backend_model_candidates(MODEL_NAME):
        return MODEL_NAME
    return model_name


def model_for_client(model_name: str) -> str:
    if MODEL_ALIAS and model_name.lower() in backend_model_candidates(MODEL_NAME):
        return MODEL_ALIAS
    return model_name


def response_content_to_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"input_text", "output_text", "text"}:
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts)

    if content is None:
        return ""

    if isinstance(content, (dict, list)):
        return json.dumps(content)

    return str(content)


def ensure_json_string(value, default="{}"):
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


async def resize_image_to_data_url(image_source):
    raw_bytes = None

    if image_source.startswith("data:"):
        header, separator, payload = image_source.partition(",")
        if separator != ",":
            raise HTTPException(status_code=400, detail="Invalid data URL for image input.")
        if ";base64" not in header:
            raise HTTPException(status_code=400, detail="Only base64 data URLs are supported for image input.")
        try:
            raw_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="Invalid base64 payload in image data URL.")
    else:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(image_source)
            response.raise_for_status()
            raw_bytes = response.content
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to fetch image URL: {exc}")

    try:
        image = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode image input.")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="Invalid image dimensions.")

    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)

    output_buffer = io.BytesIO()
    if image.mode in ("RGBA", "LA", "P"):
        if image.mode == "P":
            image = image.convert("RGBA")
        image.save(output_buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_buffer, format="JPEG", quality=85, optimize=True)
        mime = "image/jpeg"

    encoded = base64.b64encode(output_buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


async def responses_content_to_chat_content(content):
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return response_content_to_text(content)

    chat_parts = []

    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise HTTPException(
                status_code=400,
                detail=f"content[{index}] must be an object."
            )

        part_type = part.get("type")
        if not isinstance(part_type, str):
            raise HTTPException(
                status_code=400,
                detail=f"content[{index}].type must be a string."
            )

        if part_type in {"input_text", "output_text", "text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"content[{index}].text must be a string."
            )
            chat_parts.append({"type": "text", "text": text})

        elif part_type in {"input_image", "image_url"}:
            image_url = part.get("image_url") or part.get("url")
            if isinstance(image_url, str):
                resized_data_url = await resize_image_to_data_url(image_url)
                chat_parts.append({
                    "type": "image_url",
                    "image_url": {"url": resized_data_url}
                })
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                resized_data_url = await resize_image_to_data_url(image_url["url"])
                image_url_payload = dict(image_url)
                image_url_payload["url"] = resized_data_url
                chat_parts.append({
                    "type": "image_url",
                    "image_url": image_url_payload
                })
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"content[{index}].image_url must be a string or object with a url field."
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported content part type '{part_type}'."
            )

    if not chat_parts:
        return ""

    return chat_parts


async def responses_input_to_messages(response_input):
    if isinstance(response_input, str):
        return [{"role": "user", "content": response_input}]

    if not isinstance(response_input, list):
        raise HTTPException(
            status_code=400,
            detail="Responses API 'input' must be a string or an array."
        )

    messages = []

    for item in response_input:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400,
                detail="Each item in Responses API 'input' must be an object."
            )

        item_type = item.get("type")

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise HTTPException(
                    status_code=400,
                    detail="function_call_output requires a non-empty call_id."
                )
            output_text = response_content_to_text(item.get("output"))
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output_text
            })
            continue

        if item_type == "function_call":
            call_id = item.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise HTTPException(
                    status_code=400,
                    detail="function_call requires a non-empty call_id."
                )
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": ensure_json_string(item.get("arguments"))
                    }
                }]
            })
            continue

        if item_type == "message":
            role = item.get("role", "user")
            content = await responses_content_to_chat_content(item.get("content", ""))
            messages.append({"role": role, "content": content})
            continue

        if item_type is not None and item_type != "message":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input item type '{item_type}'."
            )

        role = item.get("role", "user")
        content = await responses_content_to_chat_content(item.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


def chat_completion_to_response(chat_completion, model_name):
    if not isinstance(chat_completion, dict):
        raise HTTPException(status_code=502, detail="Model runner returned an invalid chat completion body.")

    choices = chat_completion.get("choices", [])
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not isinstance(content, str):
        content = response_content_to_text(content)
    role = message.get("role", "assistant") if isinstance(message, dict) else "assistant"

    usage = chat_completion.get("usage", {}) if isinstance(chat_completion, dict) else {}

    output = []

    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            if not isinstance(function, dict):
                function = {}
            output.append({
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "call_id": tool_call.get("id", f"call_{uuid.uuid4().hex}"),
                "name": function.get("name", ""),
                "arguments": ensure_json_string(function.get("arguments")),
                "status": "completed"
            })

    if isinstance(content, str) and content:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": role,
            "content": [
                {
                    "type": "output_text",
                    "text": content,
                    "annotations": []
                }
            ]
        })

    if not output:
        output.append({
            "id": f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "status": "completed",
            "role": role,
            "content": [
                {
                    "type": "output_text",
                    "text": "",
                    "annotations": []
                }
            ]
        })

    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": model_for_client(model_name),
        "output": output,
        "output_text": content if isinstance(content, str) else str(content),
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        }
    }


def response_output_to_messages(response_obj):
    output_items = response_obj.get("output", [])
    if not isinstance(output_items, list):
        return []

    text_parts = []
    tool_calls = []
    role = "assistant"

    for item in output_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message":
            role = item.get("role", role)
            content = response_content_to_text(item.get("content", []))
            if content:
                text_parts.append(content)

        elif item_type == "function_call":
            tool_calls.append({
                "id": item.get("call_id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": ensure_json_string(item.get("arguments"))
                }
            })

    if not text_parts and not tool_calls:
        return []

    message = {
        "role": role,
        "content": "\n".join(text_parts)
    }
    if tool_calls:
        message["tool_calls"] = tool_calls

    return [message]


def normalize_responses_tool_choice(tool_choice):
    if isinstance(tool_choice, str):
        if tool_choice in {"none", "auto", "required"}:
            return tool_choice
        return tool_choice

    if not isinstance(tool_choice, dict):
        return None

    if tool_choice.get("type") == "function" and isinstance(tool_choice.get("name"), str):
        return {
            "type": "function",
            "function": {"name": tool_choice["name"]}
        }

    if (
        tool_choice.get("type") == "function"
        and isinstance(tool_choice.get("function"), dict)
        and isinstance(tool_choice["function"].get("name"), str)
    ):
        return {
            "type": "function",
            "function": {"name": tool_choice["function"]["name"]}
        }

    return tool_choice


def normalize_responses_tool(tool):
    if not isinstance(tool, dict):
        return None

    if tool.get("type") != "function":
        return None

    if isinstance(tool.get("function"), dict):
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            return None
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {"type": "object", "properties": {}})
            }
        }

    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None

    parameters = tool.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": tool.get("description", ""),
            "parameters": parameters
        }
    }


def validate_response_format(response_format):
    if not isinstance(response_format, dict):
        raise HTTPException(status_code=400, detail="response_format must be an object.")

    format_type = response_format.get("type")
    if format_type not in {"text", "json_object", "json_schema"}:
        raise HTTPException(
            status_code=400,
            detail="response_format.type must be one of: text, json_object, json_schema."
        )

    if format_type == "json_schema":
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, dict):
            raise HTTPException(
                status_code=400,
                detail="response_format.json_schema must be an object for json_schema type."
            )
        if not isinstance(json_schema.get("schema"), dict):
            raise HTTPException(
                status_code=400,
                detail="response_format.json_schema.schema must be an object."
            )


async def make_chat_payload_for_responses(payload, runner_model):
    chat_messages = await responses_input_to_messages(payload.get("input", ""))
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        chat_messages.insert(0, {"role": "system", "content": instructions})

    chat_payload = {
        "model": runner_model,
        "messages": chat_messages
    }

    previous_response_id = payload.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        previous_messages = RESPONSES_CHAT_HISTORY.get(previous_response_id)
        if previous_messages is None:
            previous_response = RESPONSES_STORE.get(previous_response_id)
            if previous_response is not None:
                previous_messages = response_output_to_messages(previous_response)
        if previous_messages is None:
            raise HTTPException(status_code=404, detail="previous_response_id not found")
        chat_payload["messages"] = previous_messages + chat_payload["messages"]

    passthrough_fields = [
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop"
    ]

    for field in passthrough_fields:
        if field in payload:
            chat_payload[field] = payload[field]

    if "max_output_tokens" in payload and "max_tokens" not in chat_payload:
        chat_payload["max_tokens"] = payload["max_output_tokens"]

    tools = payload.get("tools")
    if isinstance(tools, list):
        chat_tools = []
        for tool in tools:
            chat_tool = normalize_responses_tool(tool)
            if chat_tool is not None:
                chat_tools.append(chat_tool)
        if chat_tools:
            chat_payload["tools"] = chat_tools

    tool_choice = normalize_responses_tool_choice(payload.get("tool_choice"))
    if tool_choice is not None and chat_payload.get("tools"):
        chat_payload["tool_choice"] = tool_choice

    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        validate_response_format(response_format)
        if response_format.get("type") != "text":
            chat_payload["response_format"] = response_format

    text_config = payload.get("text")
    if isinstance(text_config, dict):
        text_format = text_config.get("format")
        if isinstance(text_format, dict):
            format_type = text_format.get("type")
            if format_type == "json_schema":
                schema = text_format.get("schema", {})
                if not isinstance(schema, dict):
                    raise HTTPException(
                        status_code=400,
                        detail="text.format.schema must be an object for json_schema."
                    )
                chat_payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": text_format.get("name", "structured_output"),
                        "schema": schema,
                        "strict": bool(text_format.get("strict", False))
                    }
                }
                validate_response_format(chat_payload["response_format"])
            elif format_type == "json_object":
                chat_payload["response_format"] = {"type": "json_object"}
                validate_response_format(chat_payload["response_format"])
            elif format_type == "text":
                pass
            else:
                raise HTTPException(
                    status_code=400,
                    detail="text.format.type must be one of: text, json_object, json_schema."
                )

    return chat_payload


def build_responses_sse_stream(response_obj):
    response_id = response_obj["id"]
    async def event_stream():
        yield f"data: {json.dumps({'type': 'response.created', 'response': response_obj})}\n\n"

        output_items = response_obj.get("output", [])
        if not isinstance(output_items, list):
            output_items = []

        for output_index, item in enumerate(output_items):
            if not isinstance(item, dict):
                continue

            yield f"data: {json.dumps({'type': 'response.output_item.added', 'response_id': response_id, 'output_index': output_index, 'item': item})}\n\n"

            if item.get("type") == "message":
                content_items = item.get("content", [])
                if isinstance(content_items, list) and content_items:
                    for content_index, part in enumerate(content_items):
                        if not isinstance(part, dict):
                            continue
                        if part.get("type") != "output_text":
                            continue

                        text = part.get("text", "")
                        yield f"data: {json.dumps({'type': 'response.content_part.added', 'response_id': response_id, 'output_index': output_index, 'content_index': content_index, 'part': {'type': 'output_text', 'text': ''}})}\n\n"
                        yield f"data: {json.dumps({'type': 'response.output_text.delta', 'response_id': response_id, 'output_index': output_index, 'content_index': content_index, 'item_id': item.get('id'), 'delta': text})}\n\n"
                        yield f"data: {json.dumps({'type': 'response.output_text.done', 'response_id': response_id, 'output_index': output_index, 'content_index': content_index, 'item_id': item.get('id'), 'text': text})}\n\n"
                        yield f"data: {json.dumps({'type': 'response.content_part.done', 'response_id': response_id, 'output_index': output_index, 'content_index': content_index, 'part': {'type': 'output_text', 'text': text}})}\n\n"

            elif item.get("type") == "function_call":
                arguments = item.get("arguments", "{}")
                yield f"data: {json.dumps({'type': 'response.function_call_arguments.delta', 'response_id': response_id, 'output_index': output_index, 'item_id': item.get('id'), 'delta': arguments})}\n\n"
                yield f"data: {json.dumps({'type': 'response.function_call_arguments.done', 'response_id': response_id, 'output_index': output_index, 'item_id': item.get('id'), 'arguments': arguments})}\n\n"

            yield f"data: {json.dumps({'type': 'response.output_item.done', 'response_id': response_id, 'output_index': output_index, 'item': item})}\n\n"

        yield f"data: {json.dumps({'type': 'response.completed', 'response': response_obj})}\n\n"
        yield "data: [DONE]\n\n"

    return event_stream()


@app.get("/health")
async def health():

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{MODEL_RUNNER_URL}/v1/models"
            )

        if response.status_code != 200:
            raise Exception("Model runner unavailable")

        return {
            "status": "ok",
            "model_runner": "reachable"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():

    try:
        async with httpx.AsyncClient(timeout=300) as client:

            response = await client.get(
                f"{MODEL_RUNNER_URL}/v1/models"
            )

        content = json_or_error_response(response)
        if isinstance(content, dict):
            models = content.get("data")
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and isinstance(model.get("id"), str):
                        model["id"] = model_for_client(model["id"])

        return JSONResponse(
            status_code=response.status_code,
            content=content
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    payload = await request_json(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    # inject model automatically if missing
    payload.setdefault("model", MODEL_NAME)
    if not isinstance(payload["model"], str):
        raise HTTPException(status_code=400, detail="model must be a string.")
    payload["model"] = model_for_runner(payload["model"])

    try:
        async with httpx.AsyncClient(timeout=300) as client:

            response = await client.post(
                f"{MODEL_RUNNER_URL}/v1/chat/completions",
                json=payload,
            )

        return JSONResponse(
            status_code=response.status_code,
            content=json_or_error_response(response)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
async def responses(request: Request):

    payload = await request_json(request)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    response_model = payload.get("model", MODEL_NAME)
    if not isinstance(response_model, str):
        raise HTTPException(status_code=400, detail="model must be a string.")
    runner_model = model_for_runner(response_model)
    chat_payload = await make_chat_payload_for_responses(payload, runner_model)
    stream = bool(payload.get("stream", False))

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{MODEL_RUNNER_URL}/v1/chat/completions",
                json=chat_payload,
            )

        content = json_or_error_response(response)
        if response.status_code >= 400:
            return JSONResponse(status_code=response.status_code, content=content)

        response_obj = chat_completion_to_response(content, response_model)
        if payload.get("store", True) is not False:
            RESPONSES_STORE[response_obj["id"]] = response_obj
            RESPONSES_CHAT_HISTORY[response_obj["id"]] = (
                chat_payload["messages"] + response_output_to_messages(response_obj)
            )

        if stream:
            return StreamingResponse(
                build_responses_sse_stream(response_obj),
                media_type="text/event-stream"
            )

        return JSONResponse(status_code=200, content=response_obj)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):

    response_obj = RESPONSES_STORE.get(response_id)
    if response_obj is None:
        raise HTTPException(status_code=404, detail="Response not found")

    return JSONResponse(status_code=200, content=response_obj)


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):

    response_obj = RESPONSES_STORE.pop(response_id, None)
    RESPONSES_CHAT_HISTORY.pop(response_id, None)
    if response_obj is None:
        raise HTTPException(status_code=404, detail="Response not found")

    return JSONResponse(
        status_code=200,
        content={
            "id": response_id,
            "object": "response.deleted",
            "deleted": True
        }
    )
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002)
