"""
nodes.py
--------
Portage ComfyUI de la partie "Story -> Sequences" de H3 Prompt Studio
(cf. sequence_pipeline.py / llm_client.py fournis par l'app d'origine).

Un seul node : H3StoryToSequences
  Entrées : 9 slots IMAGE individuels (image_1 requis, image_2..image_9
  optionnels), chacun branché depuis son propre loader/node en amont, avec
  un rôle texte optionnel par image.
  Sorties : "story" (texte complet de l'histoire, en STRING simple — reliez
            un node "Show Text"/"Display Text" dessus pour le lire) + 10
            sorties STRING numérotées (prompt_1..prompt_10), une par
            séquence. Les sorties au-delà de n_sequences sont une chaîne
            vide "".

Trois passes LLM (identiques à l'appli d'origine) :
  Pass A - un LLM "vision" décrit chaque image, puis un LLM "story" écrit
           une histoire courte à partir de ces descriptions + une prémisse.
  Pass B - un LLM "sequence" découpe l'histoire en N séquences (JSON).
  Pass C - le même LLM "sequence" développe chaque séquence (via
           sequence_to_brief -> REF2VA_SYSTEM_PROMPT) en un prompt H3
           Ref2VA final (6 sections), distribué vers la sortie numérotée.
"""

import io
import os
import tempfile
import time

import numpy as np
from PIL import Image

from .llm_client import LLMClient, LLMError
from .sequence_pipeline import (
    generate_story,
    generate_sequence_breakdown,
    sequence_to_brief,
    _reference_library_text,
    _sequence_max_tokens,
)
from .system_prompts import REF2VA_SYSTEM_PROMPT
# Source unique des hosts par défaut par backend (partagée avec api.py et,
# via la route /h3_prompt_studio/models, avec h3_story_sequences.js) : évite
# que les trois couches divergent si un host par défaut change un jour.
from .api import DEFAULT_HOSTS

# TTL cache for the model probe below: INPUT_TYPES is a classmethod that
# ComfyUI can re-evaluate on every graph load/refresh, so a naive network
# call there would block the UI whenever the LLM server is slow or down.
_MODEL_PROBE_CACHE: dict = {}
_MODEL_PROBE_TTL_SECONDS = 30

MAX_SEQUENCES = 10
MAX_IMAGES = 9

STYLE_OPTIONS = [
    "Cinematic", "live-action", "2D-animated", "3D CG", "claymation",
    "watercolor", "vintage film", "auto (laisser le LLM choisir)",
]

CAMERA_MOTIONS = [
    "Zoom In", "Zoom Out", "Push In", "Pull Out",
    "Pan Left", "Pan Right", "Truck Left", "Truck Right",
    "Tilt Up", "Tilt Down", "Pedestal Up", "Pedestal Down",
    "Arc Shot", "Tracking Shot", "Static Shot",
    "Shake Slightly", "Shake Strongly", "POV",
    "Roll Clockwise", "Roll Counterclockwise",
]

LANGUAGES = [
    "English", "French", "Chinese", "Spanish", "German", "Italian",
    "Japanese", "Korean", "Portuguese", "Russian", "Arabic",
]

_FALLBACK_MODELS = ["(saisir un nom de modele)"]


def _probe_ollama_models(host: str = None) -> list:
    """Best-effort discovery for the model combos. Never raises: ComfyUI calls
    INPUT_TYPES at graph-definition time and a dead/absent LLM server must not
    break node loading. Cached for _MODEL_PROBE_TTL_SECONDS so repeated
    graph loads/refreshes don't re-hit the network every time."""
    host = host or DEFAULT_HOSTS["ollama"]
    now = time.monotonic()
    cached = _MODEL_PROBE_CACHE.get(host)
    if cached and (now - cached[0]) < _MODEL_PROBE_TTL_SECONDS:
        return cached[1]
    try:
        client = LLMClient(base_url=host, backend="ollama", timeout=3)
        models = [m.name for m in client.list_models() if m.name]
        result = models or list(_FALLBACK_MODELS)
    except Exception:
        result = list(_FALLBACK_MODELS)
    _MODEL_PROBE_CACHE[host] = (now, result)
    return result


def _tensor_to_tempfile(img_tensor) -> str:
    """Converts one HWC float[0,1] image tensor (a single frame taken from a
    ComfyUI IMAGE input) to a temporary PNG file and returns its path."""
    arr = img_tensor.cpu().numpy() if hasattr(img_tensor, "cpu") else np.array(img_tensor)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr)
    fd, path = tempfile.mkstemp(suffix=".png", prefix="h3_ref_")
    os.close(fd)
    pil_img.save(path)
    return path


def _first_frame(image_input):
    """A ComfyUI IMAGE input is a [B,H,W,C] tensor even for a single image
    (B=1 in that case). Returns the first frame as an HWC tensor/array."""
    return image_input[0]


REF2VA_MAX_TOKENS = 1500
REF2VA_RETRY_MAX_TOKENS = 3000
REF2VA_REQUIRED_SECTIONS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _missing_ref2va_sections(text: str) -> list:
    """Detects a Pass C output truncated mid-way (sequences 4-5 issue): if the
    model runs out of max_tokens before finishing the 6 mandatory sections,
    the trailing ones are simply absent from the text. Cheap substring check
    - good enough to trigger a retry with a bigger token budget."""
    lowered = (text or "").lower()
    return [s for s in REF2VA_REQUIRED_SECTIONS if s not in lowered]


class H3StoryToSequences:
    """Story -> Sequences : references images -> story -> N prompt Ref2VA."""

    @classmethod
    def INPUT_TYPES(cls):
        models = _probe_ollama_models()
        return {
            "required": {
                "image_1": ("IMAGE",),
                "backend": (["ollama", "lmstudio", "llamacpp"], {"default": "ollama"}),
                "host": ("STRING", {"default": DEFAULT_HOSTS["ollama"]}),
                "vision_model": (models, {"default": models[0]}),
                "story_model": (models, {"default": models[0]}),
                "sequence_model": (models, {"default": models[0]}),
                "premise": ("STRING", {"multiline": True, "default": ""}),
                "language": (LANGUAGES, {"default": "English"}),
                "style": (STYLE_OPTIONS, {"default": "Cinematic"}),
                "word_count": ("INT", {"default": 350, "min": 50, "max": 2000}),
                "n_sequences": ("INT", {"default": 10, "min": 1, "max": MAX_SEQUENCES}),
                "duration_per_sequence": ("INT", {"default": 8, "min": 1, "max": 60}),
                "num_ctx": ("INT", {
                    "default": 32768, "min": 2048, "max": 131072,
                    "tooltip": (
                        "Fenêtre de contexte demandée au backend (Ollama : appliquée directement. "
                        "LM Studio/llama.cpp : ce champ est ignoré côté API OpenAI-compatible — réglez "
                        "'Context Length' dans LM Studio au chargement du modèle, sinon la Pass C peut "
                        "recevoir un prompt tronqué et renvoyer une réponse vide.)"
                    ),
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "temperature_story": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature_sequence": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "temperature_prompt": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                **{f"image_{i}": ("IMAGE",) for i in range(2, MAX_IMAGES + 1)},
                **{
                    f"role_{i}": ("STRING", {
                        "default": "", "tooltip": f"Rôle de l'image {i} (facultatif).",
                    })
                    for i in range(1, MAX_IMAGES + 1)
                },
                "extra_instructions": ("STRING", {"multiline": True, "default": ""}),
                "video_music": ("STRING", {
                    "multiline": False, "default": "",
                    "tooltip": "Note de musique pour toute la vidéo, répétée identique dans chaque séquence (facultatif).",
                }),
                "no_video_music": ("BOOLEAN", {"default": False}),
                "camera_motions": ("STRING", {
                    "multiline": True, "default": ", ".join(CAMERA_MOTIONS),
                }),
            },
        }

    RETURN_TYPES = ("STRING",) + ("STRING",) * MAX_SEQUENCES
    RETURN_NAMES = ("story",) + tuple(f"prompt_{i}" for i in range(1, MAX_SEQUENCES + 1))
    FUNCTION = "run"
    CATEGORY = "H3 Prompt Studio"

    @classmethod
    def VALIDATE_INPUTS(cls, vision_model, story_model, sequence_model):
        # Ces 3 combos sont sondés côté Ollama une seule fois, au chargement du
        # node (cf. _probe_ollama_models() dans INPUT_TYPES). La vraie liste de
        # modèles est ensuite rafraîchie dynamiquement côté JS selon le backend
        # choisi (web/h3_story_sequences.js + api.py) : leur valeur réelle
        # n'apparaît donc jamais dans cette liste figée. On désactive ici le
        # contrôle "valeur présente dans la liste" pour ces 3 champs -
        # LLMClient renverra de toute façon une erreur claire à l'exécution si
        # le modèle n'existe pas vraiment sur le backend visé.
        return True

    def run(
        self,
        image_1,
        backend,
        host,
        vision_model,
        story_model,
        sequence_model,
        premise,
        language,
        style,
        word_count,
        n_sequences,
        duration_per_sequence,
        num_ctx,
        temperature_story,
        temperature_sequence,
        temperature_prompt,
        seed,
        image_2=None, image_3=None, image_4=None, image_5=None,
        image_6=None, image_7=None, image_8=None, image_9=None,
        role_1="", role_2="", role_3="", role_4="", role_5="",
        role_6="", role_7="", role_8="", role_9="",
        extra_instructions="",
        video_music="",
        no_video_music=False,
        camera_motions="",
    ):
        client = LLMClient(base_url=host, backend=backend)

        all_images = [image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]
        all_roles = [role_1, role_2, role_3, role_4, role_5, role_6, role_7, role_8, role_9]

        # --- description de chaque image de reference (LLM vision) --------
        references = []
        tmp_paths = []
        try:
            for i, img in enumerate(all_images):
                if img is None:
                    continue
                path = _tensor_to_tempfile(_first_frame(img))
                tmp_paths.append(path)
                try:
                    description = client.describe_image(vision_model, path)
                except LLMError as e:
                    description = f"(description indisponible : {e})"
                role = all_roles[i].strip() if all_roles[i] and all_roles[i].strip() else f"Reference {i + 1}"
                references.append({"type": "Picture", "role": role, "description": description})
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # --- Pass A : histoire ---------------------------------------------
        story_text = generate_story(
            client=client,
            model=story_model,
            references=references,
            premise=premise,
            language=language,
            word_count=word_count,
            temperature=temperature_story,
            num_ctx=num_ctx,
            seed=seed,
        )

        # --- Pass B : decoupage en sequences ---------------------------------
        motions = [m.strip() for m in (camera_motions or "").split(",") if m.strip()] or CAMERA_MOTIONS
        sequences = generate_sequence_breakdown(
            client=client,
            model=sequence_model,
            story_text=story_text,
            references=references,
            n_sequences=n_sequences,
            camera_motions=motions,
            extra_instructions=extra_instructions,
            duration_per_sequence=duration_per_sequence,
            temperature=temperature_sequence,
            num_ctx=num_ctx,
            seed=seed,
        )

        ref_block = _reference_library_text(references)
        if no_video_music:
            music_line = (
                "MUSIC FOR THE WHOLE VIDEO: explicitly none, non_diegetic_music must be "
                "N/A (keep identical across every scene)."
            )
        elif video_music.strip():
            music_line = f"MUSIC FOR THE WHOLE VIDEO (keep identical across every scene): {video_music.strip()}"
        else:
            music_line = ""

        prompts = [""] * MAX_SEQUENCES
        for seq in sequences:
            idx = seq.get("index", 0)
            if not (1 <= idx <= MAX_SEQUENCES):
                continue
            brief = sequence_to_brief(
                seq=seq,
                n_total=n_sequences,
                reference_library_block=ref_block,
                overall_story=story_text,
                duration_per_sequence=duration_per_sequence,
                style=style,
                extra_instructions=extra_instructions,
                video_music=music_line,
            )
            # --- Pass C : le "brief" n'est qu'un prompt intermediaire -
            # c'est cette 3e passe (system prompt REF2VA_SYSTEM_PROMPT) qui
            # produit le prompt H3 Ref2VA final en 6 sections.
            final_prompt = client.chat(
                model=sequence_model,
                system_prompt=REF2VA_SYSTEM_PROMPT,
                user_prompt=brief,
                temperature=temperature_prompt,
                num_ctx=num_ctx,
                max_tokens=REF2VA_MAX_TOKENS,
                seed=seed,
                think=False,  # cf. llm_client.LLMClient.chat() : evite qu'un modele
                # "thinking" (qwen3.x, deepseek-r1...) ne brule tout le budget
                # max_tokens en raisonnement cache avant meme d'ecrire les 6
                # sections attendues.
            )
            missing = _missing_ref2va_sections(final_prompt)
            if missing:
                # Sortie coupée avant la fin des 6 sections (observé sur des
                # scènes plus denses en personnages/dialogue) : un seul
                # retry avec un budget nettement plus large, même mécanisme
                # de filet de sécurité que Pass B.
                final_prompt = client.chat(
                    model=sequence_model,
                    system_prompt=REF2VA_SYSTEM_PROMPT,
                    user_prompt=brief,
                    temperature=temperature_prompt,
                    num_ctx=num_ctx,
                    max_tokens=REF2VA_RETRY_MAX_TOKENS,
                    seed=seed,
                    think=False,
                )
                still_missing = _missing_ref2va_sections(final_prompt)
                if still_missing:
                    final_prompt += (
                        f"\n\n[AVERTISSEMENT H3 Prompt Studio : sortie probablement tronquée - "
                        f"sections manquantes : {', '.join(still_missing)}. Augmentez num_ctx et/ou "
                        f"REF2VA_RETRY_MAX_TOKENS dans nodes.py si cela se reproduit.]"
                    )
            prompts[idx - 1] = final_prompt

        return (story_text, *prompts)


NODE_CLASS_MAPPINGS = {
    "H3StoryToSequences": H3StoryToSequences,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3StoryToSequences": "H3 Story -> Sequences (9 refs -> 10 prompts)",
}
