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

app = FastAPI(title="Humatheque VLM Proxy")


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

        return JSONResponse(
            status_code=response.status_code,
            content=response.json()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    payload = await request.json()

    # inject model automatically if missing
    payload.setdefault("model", MODEL_NAME)

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