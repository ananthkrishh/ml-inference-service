"""POST /generate — text generation via Llama 3 on Ollama. Implemented in step 3."""

from fastapi import APIRouter

router = APIRouter(tags=["llm"])

# TODO (step 3):
#   - accept {prompt, max_tokens?}
#   - call app/ollama_client.py, return text + timing/token stats
#   - map Ollama timeouts/unavailability to 504/503
