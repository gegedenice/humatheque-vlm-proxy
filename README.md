# Humatheque VLM Proxy

FastAPI proxy that exposes OpenAI-compatible routes in front of a Docker model runner.

It supports:
- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses` (OpenAI Responses-like route)
- `GET /v1/responses/{response_id}`
- `DELETE /v1/responses/{response_id}`

## 1) Start the backend model

Use these commands to pull, configure, and run the backend model:

```bash
docker model pull hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL
docker model configure --context-size 16384 hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL -- --temp 0.7 --top-p 0.9 --top-k 20 --repeat-penalty 1.05 --threads 8 --mlock --batch-size 512
docker model run hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL
```

Default backend URL expected by this proxy:
- `http://host.docker.internal:12434`

## 2) Configure the proxy

Environment variables:
- `VLM_RUNNER_URL`: backend model runner URL (default: `http://host.docker.internal:12434`)
- `VLM_NAME`: backend model ID used by runner (default: `hf.co/unsloth/Qwen3-VL-8B-Instruct-GGUF:UD-Q4_K_XL`)
- `VLM_ALIAS`: friendly model alias exposed to clients (default: `Qwen3-VL-8B-Instruct-GGUF`)

### Why alias?

Clients usually call `GET /v1/models` and then reuse the returned model name for generation.  
This proxy maps backend model IDs to `VLM_ALIAS` for clients and maps that alias back to `VLM_NAME` for backend calls.

## 3) Install and run proxy

```bash
pip install -r requirements.txt
python app.py
```

The API is served on:
- `http://0.0.0.0:8002`

## 4) API examples

### List models

```bash
curl http://localhost:8002/v1/models
```

### Chat Completions

```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-8B-Instruct-GGUF",
    "messages": [
      {"role":"user","content":"Hello!"}
    ]
  }'
```

### Responses API-like route

```bash
curl -X POST http://localhost:8002/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-VL-8B-Instruct-GGUF",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type":"input_text","text":"Describe this image briefly"},
          {"type":"input_image","image_url":"https://example.com/image.jpg"}
        ]
      }
    ]
  }'
```

## Notes

- `POST /v1/responses` includes support for tool-calling style loops (`tools`, `tool_choice`, `function_call_output`, `previous_response_id`).
- Response IDs are stored in-memory for `GET`/`DELETE` routes (data is lost on process restart).
