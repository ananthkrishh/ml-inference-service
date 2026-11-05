"""POST /generate — text generation via Llama 3 running in Ollama."""

import time

from fastapi import APIRouter, HTTPException, Request

from app.metrics import INFERENCE_ERRORS, LLM_TOKENS, LLM_TOKENS_PER_SECOND, MODEL_INFERENCE
from app.ollama_client import OllamaError, OllamaTimeout, OllamaUnavailable
from app.schemas import GenerateRequest, GenerateResponse

router = APIRouter(tags=["llm"])


# `async def` (unlike /predict): this endpoint mostly WAITS on Ollama over the network,
# so while it waits the event loop can serve other requests.
@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    ollama = request.app.state.ollama
    start = time.perf_counter()
    try:
        result = await ollama.generate(
            body.prompt, max_tokens=body.max_tokens, temperature=body.temperature
        )
    except OllamaTimeout as exc:
        INFERENCE_ERRORS.labels(model="llm", reason="timeout").inc()
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except OllamaUnavailable as exc:
        INFERENCE_ERRORS.labels(model="llm", reason="unavailable").inc()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaError as exc:
        INFERENCE_ERRORS.labels(model="llm", reason="bad_response").inc()
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    MODEL_INFERENCE.labels(model="llm").observe(time.perf_counter() - start)
    LLM_TOKENS.labels(kind="prompt").inc(result.prompt_tokens)
    LLM_TOKENS.labels(kind="completion").inc(result.completion_tokens)
    if result.tokens_per_second > 0:
        LLM_TOKENS_PER_SECOND.observe(result.tokens_per_second)

    return GenerateResponse(
        text=result.text,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_seconds=result.total_seconds,
        load_seconds=result.load_seconds,
        tokens_per_second=result.tokens_per_second,
    )
