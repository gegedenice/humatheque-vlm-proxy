import os
import json
import time
import uuid
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

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

app = FastAPI(title="Humatheque VLM Proxy")
RESPONSES_STORE = {}


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


def responses_content_to_chat_content(content):
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
            image_url = part.get("image_url")
            if isinstance(image_url, str):
                chat_parts.append({
                    "type": "image_url",
                    "image_url": {"url": image_url}
                })
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                chat_parts.append({
                    "type": "image_url",
                    "image_url": image_url
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


def responses_input_to_messages(response_input):
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

        if item_type == "message":
            role = item.get("role", "user")
            content = responses_content_to_chat_content(item.get("content", ""))
            messages.append({"role": role, "content": content})
            continue

        if item_type is not None and item_type != "message":
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported input item type '{item_type}'."
            )

        role = item.get("role", "user")
        content = responses_content_to_chat_content(item.get("content", ""))
        messages.append({"role": role, "content": content})

    return messages


def chat_completion_to_response(chat_completion, model_name):
    choices = chat_completion.get("choices", [])
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    role = message.get("role", "assistant") if isinstance(message, dict) else "assistant"

    usage = chat_completion.get("usage", {}) if isinstance(chat_completion, dict) else {}

    output = []

    tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function", {})
            output.append({
                "id": f"fc_{uuid.uuid4().hex}",
                "type": "function_call",
                "call_id": tool_call.get("id", f"call_{uuid.uuid4().hex}"),
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
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

    messages = []
    pending_tool_calls = []

    for item in output_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        if item_type == "message":
            role = item.get("role", "assistant")
            content = response_content_to_text(item.get("content", []))
            messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            pending_tool_calls.append({
                "id": item.get("call_id", f"call_{uuid.uuid4().hex}"),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}")
                }
            })

    if pending_tool_calls:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": pending_tool_calls
        })

    return messages


def normalize_responses_tool_choice(tool_choice):
    if isinstance(tool_choice, str):
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


def make_chat_payload_for_responses(payload, runner_model):
    chat_messages = responses_input_to_messages(payload.get("input", ""))
    instructions = payload.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        chat_messages.insert(0, {"role": "system", "content": instructions})

    chat_payload = {
        "model": runner_model,
        "messages": chat_messages
    }

    previous_response_id = payload.get("previous_response_id")
    if isinstance(previous_response_id, str) and previous_response_id:
        previous_response = RESPONSES_STORE.get(previous_response_id)
        if previous_response is None:
            raise HTTPException(status_code=404, detail="previous_response_id not found")
        previous_messages = response_output_to_messages(previous_response)
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

    tools = payload.get("tools")
    if isinstance(tools, list):
        chat_tools = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("type") == "function" and isinstance(tool.get("name"), str):
                chat_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}})
                    }
                })
            elif tool.get("type") == "function" and isinstance(tool.get("function"), dict):
                chat_tools.append(tool)
        if chat_tools:
            chat_payload["tools"] = chat_tools

    tool_choice = normalize_responses_tool_choice(payload.get("tool_choice"))
    if tool_choice is not None:
        chat_payload["tool_choice"] = tool_choice

    if "parallel_tool_calls" in payload:
        chat_payload["parallel_tool_calls"] = payload["parallel_tool_calls"]

    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        validate_response_format(response_format)
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
                chat_payload["response_format"] = {"type": "text"}
                validate_response_format(chat_payload["response_format"])
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
        async with httpx.AsyncClient(timeout=None) as client:

            response = await client.get(
                f"{MODEL_RUNNER_URL}/v1/models"
            )

        content = response.json()
        if isinstance(content, dict):
            models = content.get("data")
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and "id" in model:
                        model["id"] = model_for_client(model["id"])

        return JSONResponse(
            status_code=response.status_code,
            content=content
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    payload = await request.json()

    # inject model automatically if missing
    payload.setdefault("model", MODEL_NAME)
    payload["model"] = model_for_runner(payload["model"])

    try:
        async with httpx.AsyncClient(timeout=None) as client:

            response = await client.post(
                f"{MODEL_RUNNER_URL}/v1/chat/completions",
                json=payload,
            )

        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/responses")
async def responses(request: Request):

    payload = await request.json()

    response_model = payload.get("model", MODEL_NAME)
    runner_model = model_for_runner(response_model)
    chat_payload = make_chat_payload_for_responses(payload, runner_model)
    stream = bool(payload.get("stream", False))

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"{MODEL_RUNNER_URL}/v1/chat/completions",
                json=chat_payload,
            )

        content = response.json()
        if response.status_code >= 400:
            return JSONResponse(status_code=response.status_code, content=content)

        response_obj = chat_completion_to_response(content, response_model)
        if payload.get("store", True) is not False:
            RESPONSES_STORE[response_obj["id"]] = response_obj

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