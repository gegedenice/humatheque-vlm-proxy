import os
import httpx

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

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


def model_for_runner(model_name: str) -> str:
    if MODEL_ALIAS and model_name == MODEL_ALIAS:
        return MODEL_NAME
    return model_name


def model_for_client(model_name: str) -> str:
    if MODEL_ALIAS and model_name == MODEL_NAME:
        return MODEL_ALIAS
    return model_name


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
    
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8002)