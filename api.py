"""
api.py
------
Route HTTP côté serveur ComfyUI, utilisée par web/h3_story_sequences.js pour
rafraîchir dynamiquement la liste des modèles quand l'utilisateur change de
backend ou de host dans le node H3StoryToSequences.

GET /h3_prompt_studio/models?backend=ollama|lmstudio|llamacpp&host=...
  -> {"models": [...], "default_host": "..."}
"""

from aiohttp import web
from server import PromptServer

from .llm_client import LLMClient, LLMError

DEFAULT_HOSTS = {
    "ollama": "http://localhost:11434",
    "lmstudio": "http://localhost:1234",
    "llamacpp": "http://localhost:8080",
}


@PromptServer.instance.routes.get("/h3_prompt_studio/models")
async def h3_list_models(request):
    backend = request.rel_url.query.get("backend", "ollama")
    default_host = DEFAULT_HOSTS.get(backend, DEFAULT_HOSTS["ollama"])
    host = request.rel_url.query.get("host") or default_host

    try:
        client = LLMClient(base_url=host, backend=backend)
        models = [m.name for m in client.list_models() if m.name]
    except LLMError:
        models = []
    except Exception:
        # Ne jamais laisser une erreur réseau/serveur casser l'UI ComfyUI.
        models = []

    return web.json_response({"models": models, "default_host": default_host})
