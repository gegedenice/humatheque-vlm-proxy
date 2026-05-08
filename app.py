import base64
import binascii
import io
import json
import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image

load_dotenv()

MODEL_RUNNER_URL = os.environ.get(
    "VLM_RUNNER_URL",
    "http://host.docker.internal:12434",
)

MODEL_NAME = os.environ.get(
    "VLM_NAME",
    "hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL",
)

MODEL_ALIAS = os.environ.get("VLM_ALIAS", "Qwen3-VL-8B-Instruct-GGUF")
MAX_IMAGE_SIDE = int(os.environ.get("VLM_MAX_IMAGE_SIDE", "1024"))

app = FastAPI(title="Humatheque VLM Proxy")


def json_or_error_response(response, upstream_name="model_runner"):
    try:
        return response.json()
    except json.JSONDecodeError:
        message = response.text.strip() or response.reason_phrase
        return {
            "error": {
                "message": message,
                "type": f"{upstream_name}_error",
                "status_code": response.status_code,
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


def ensure_json_string(value, default="{}"):
    if isinstance(value, str):
        return value
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


async def resize_image_to_data_url(image_source):
    if not isinstance(image_source, str) or not image_source:
        raise HTTPException(status_code=400, detail="Image URL must be a non-empty string.")

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


async def normalize_chat_content(content):
    if isinstance(content, str) or content is None:
        return content

    if not isinstance(content, list):
        raise HTTPException(status_code=400, detail="message.content must be a string, null, or an array.")

    chat_parts = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise HTTPException(status_code=400, detail=f"message.content[{index}] must be an object.")

        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            text = part.get("text")
            if not isinstance(text, str):
                raise HTTPException(status_code=400, detail=f"message.content[{index}].text must be a string.")
            chat_parts.append({"type": "text", "text": text})
            continue

        if part_type in {"image_url", "input_image"}:
            image_url = part.get("image_url") or part.get("url")
            image_payload = {}
            if isinstance(image_url, str):
                image_payload["url"] = image_url
            elif isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                image_payload = dict(image_url)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"message.content[{index}].image_url must be a string or object with a url field.",
                )
            image_payload["url"] = await resize_image_to_data_url(image_payload["url"])
            chat_parts.append({"type": "image_url", "image_url": image_payload})
            continue

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported message content part type '{part_type}'.",
        )

    return chat_parts


def normalize_tool_calls(tool_calls):
    if not isinstance(tool_calls, list):
        return tool_calls

    normalized = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            normalized.append(tool_call)
            continue

        normalized_tool_call = dict(tool_call)
        function = normalized_tool_call.get("function")
        if isinstance(function, dict) and "arguments" in function:
            normalized_function = dict(function)
            normalized_function["arguments"] = ensure_json_string(function.get("arguments"))
            normalized_tool_call["function"] = normalized_function
        normalized.append(normalized_tool_call)

    return normalized


async def normalize_chat_payload(payload):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")

    payload = dict(payload)
    payload.setdefault("model", MODEL_NAME)
    if not isinstance(payload["model"], str):
        raise HTTPException(status_code=400, detail="model must be a string.")
    payload["model"] = model_for_runner(payload["model"])

    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array.")

    normalized_messages = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise HTTPException(status_code=400, detail=f"messages[{index}] must be an object.")

        normalized_message = dict(message)
        role = normalized_message.get("role")
        if not isinstance(role, str):
            raise HTTPException(status_code=400, detail=f"messages[{index}].role must be a string.")

        if "content" in normalized_message:
            normalized_message["content"] = await normalize_chat_content(normalized_message["content"])

        if "tool_calls" in normalized_message:
            normalized_message["tool_calls"] = normalize_tool_calls(normalized_message["tool_calls"])

        normalized_messages.append(normalized_message)

    payload["messages"] = normalized_messages
    return payload


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{MODEL_RUNNER_URL}/v1/models")

        if response.status_code != 200:
            raise Exception("Model runner unavailable")

        return {
            "status": "ok",
            "model_runner": "reachable",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.get(f"{MODEL_RUNNER_URL}/v1/models")

        content = json_or_error_response(response)
        if isinstance(content, dict):
            models = content.get("data")
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict) and isinstance(model.get("id"), str):
                        model["id"] = model_for_client(model["id"])

        return JSONResponse(
            status_code=response.status_code,
            content=content,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    payload = await normalize_chat_payload(await request_json(request))

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{MODEL_RUNNER_URL}/v1/chat/completions",
                json=payload,
            )

        return JSONResponse(
            status_code=response.status_code,
            content=json_or_error_response(response),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002)
