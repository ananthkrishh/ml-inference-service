"""POST /generate — text generation via Llama 3 running in Ollama."""

from fastapi import APIRouter, HTTPException, Request

from app.ollama_client import OllamaError, OllamaTimeout, OllamaUnavailable
from app.schemas import GenerateRequest, GenerateResponse

router = APIRouter(tags=["llm"])


# `async def` (unlike /predict): this endpoint mostly WAITS on Ollama over the network,
# so while it waits the event loop can serve other requests.
@router.post("/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    ollama = request.app.state.ollama
    try:
        result = await ollama.generate(
            body.prompt, max_tokens=body.max_tokens, temperature=body.temperature
        )
    except OllamaTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GenerateResponse(
        text=result.text,
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        total_seconds=result.total_seconds,
        load_seconds=result.load_seconds,
        tokens_per_second=result.tokens_per_second,
    )
