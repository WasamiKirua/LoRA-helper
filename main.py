from __future__ import annotations

import io
import json
import zipfile
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from captioning import (
    build_ai_prompt,
    build_final_caption,
    generate_gemini_caption,
    generate_grok_caption,
)
from config import CONFIG, TOOLTIPS, get_effective_config

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ENV_PATH = BASE_DIR / ".env"
METADATA_PATH = BASE_DIR / "metadata.json"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory="templates")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    STATE["logs"].insert(0, f"[{timestamp}] {message}")


def all_done(images: list[dict]) -> bool:
    return bool(images) and all(img["status"] in {"done", "error"} for img in images)


def can_process(images: list[dict]) -> bool:
    return bool(images) and any(img["status"] in {"pending", "error"} for img in images)


def next_pending_image_index(queue: list[str], start_index: int, images: list[dict]) -> int:
    lookup = {image["id"]: image for image in images}
    for idx in range(start_index, len(queue)):
        image = lookup.get(queue[idx])
        if image and image["status"] in {"pending", "error"}:
            return idx
    return len(queue)


def load_env_file() -> tuple[dict[str, str], bool]:
    if not ENV_PATH.exists():
        return {}, False
    values: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip()
    return values, True


def update_env_file(updates: dict[str, str]) -> None:
    existing_lines = []
    if ENV_PATH.exists():
        existing_lines = ENV_PATH.read_text().splitlines()

    remaining = {key: value for key, value in updates.items()}
    output_lines = []
    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            output_lines.append(line)
            continue
        key, _ = stripped.split("=", 1)
        key = key.strip()
        if key in remaining:
            value = remaining.pop(key)
            if value:
                output_lines.append(f"{key}={value}")
            continue
        output_lines.append(line)

    for key, value in remaining.items():
        if value:
            output_lines.append(f"{key}={value}")

    if output_lines:
        ENV_PATH.write_text("\n".join(output_lines).strip() + "\n")


ENV_VALUES, _ = load_env_file()

STATE = {
    "images": [],
    "logs": [],
    "selected_checkpoint": "SDXL",
    "selected_use_case": "identity",
    "backend": "gemini",
    "gemini_api_key": ENV_VALUES.get("GEMINI_API_KEY", ""),
    "grok_api_key": ENV_VALUES.get("GROK_API_KEY", ""),
    "grok_model": ENV_VALUES.get("GROK_MODEL", "grok-2-vision-latest"),
    "allow_nsfw": True,
    "settings": get_effective_config("SDXL", "identity"),
    "is_processing": False,
    "processing_queue": [],
    "processing_index": 0,
    "processing_delay_remaining": 0.0,
    "simulate_failures": False,
    "master_prompt": "",
    "file_prefix": "image",
}


def load_metadata() -> list[dict]:
    if not METADATA_PATH.exists():
        return []
    try:
        payload = json.loads(METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return []
    images = payload.get("images", [])
    if not isinstance(images, list):
        return []
    restored = []
    for image in images:
        if not isinstance(image, dict):
            continue
        stored_name = image.get("stored_name")
        if not stored_name:
            continue
        file_path = UPLOAD_DIR / stored_name
        if not file_path.exists():
            continue
        if image.get("status") == "processing":
            image["status"] = "pending"
            image["progress"] = 0
        caption_txt = image.get("caption_txt")
        if caption_txt and not (UPLOAD_DIR / caption_txt).exists():
            image.pop("caption_txt", None)
        restored.append(image)
    return restored


def save_metadata() -> None:
    payload = {"images": STATE["images"]}
    METADATA_PATH.write_text(json.dumps(payload, indent=2))


STATE["images"] = load_metadata()


def resolve_active_api_key() -> str:
    if STATE["backend"] == "grok":
        return STATE.get("grok_api_key", "")
    return STATE.get("gemini_api_key", "")


def api_key_missing() -> bool:
    return not resolve_active_api_key()


def should_fail_image(image: dict) -> bool:
    if not STATE.get("simulate_failures"):
        return False
    image_id = str(image.get("id", ""))
    return sum(ord(ch) for ch in image_id) % 4 == 0


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    _, env_present = load_env_file()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "images": STATE["images"],
            "logs": STATE["logs"],
            "settings": STATE["settings"],
            "selected_checkpoint": STATE["selected_checkpoint"],
            "selected_use_case": STATE["selected_use_case"],
            "backend": STATE["backend"],
            "gemini_api_key": STATE["gemini_api_key"],
            "grok_api_key": STATE["grok_api_key"],
            "grok_model": STATE["grok_model"],
            "allow_nsfw": STATE["allow_nsfw"],
            "simulate_failures": STATE["simulate_failures"],
            "tooltips": TOOLTIPS,
            "api_key_missing": api_key_missing(),
            "env_present": env_present,
            "checkpoint_order": CONFIG["ui_hints"]["checkpoint_order"],
            "use_case_order": CONFIG["ui_hints"]["use_case_order"],
            "all_done": all_done(STATE["images"]),
            "can_process": can_process(STATE["images"]),
            "is_processing": STATE["is_processing"],
        },
    )


@app.post("/settings", response_class=HTMLResponse)
async def update_settings(request: Request) -> HTMLResponse:
    form = await request.form()
    selected_use_case = form.get("training_goal", STATE["selected_use_case"])
    selected_checkpoint = form.get("target_base_model", STATE["selected_checkpoint"])
    previous_backend = STATE["backend"]
    previous_checkpoint = STATE["selected_checkpoint"]
    previous_use_case = STATE["selected_use_case"]

    selection_changed = (
        selected_checkpoint != previous_checkpoint or selected_use_case != previous_use_case
    )

    if selection_changed:
        STATE["settings"] = get_effective_config(selected_checkpoint, selected_use_case)
        STATE["selected_checkpoint"] = selected_checkpoint
        STATE["selected_use_case"] = selected_use_case
        log(
            f"Switched context to Checkpoint: {selected_checkpoint}, Use Case: {selected_use_case}"
        )

    lora = STATE["settings"].setdefault("lora", {})
    lora["filePrefix"] = form.get("file_prefix", lora.get("filePrefix", ""))
    lora["keyword"] = form.get("keyword", lora.get("keyword", ""))
    caption = STATE["settings"].setdefault("captionControl", {})
    caption["subjectToken"] = form.get("subject_token", caption.get("subjectToken", ""))

    env_values, env_present = load_env_file()
    if env_present:
        STATE["gemini_api_key"] = env_values.get("GEMINI_API_KEY", STATE["gemini_api_key"])
        STATE["grok_api_key"] = env_values.get("GROK_API_KEY", STATE["grok_api_key"])
        STATE["grok_model"] = env_values.get("GROK_MODEL", STATE["grok_model"])

    STATE["backend"] = form.get("backend", STATE["backend"])
    STATE["gemini_api_key"] = form.get("gemini_api_key", STATE["gemini_api_key"])
    STATE["grok_api_key"] = form.get("grok_api_key", STATE["grok_api_key"])
    STATE["grok_model"] = form.get("grok_model", STATE["grok_model"])
    STATE["allow_nsfw"] = "allow_nsfw" in form
    STATE["simulate_failures"] = "simulate_failures" in form

    update_env_file(
        {
            "GEMINI_API_KEY": STATE["gemini_api_key"],
            "GROK_API_KEY": STATE["grok_api_key"],
            "GROK_MODEL": STATE["grok_model"],
        }
    )

    if STATE["backend"] != previous_backend:
        backend_label = "Grok (xAI)" if STATE["backend"] == "grok" else "Gemini"
        log(f"Captioning backend set to {backend_label}.")

    if not selection_changed:
        caption["guidance"] = form.get("caption_guidance", caption.get("guidance", ""))
        caption["guidanceStrength"] = float(
            form.get("guidance_strength", caption.get("guidanceStrength", 0.7))
        )
        caption["negativeHints"] = form.get(
            "negative_hints", caption.get("negativeHints", "")
        )
        caption["captionLength"] = form.get(
            "caption_length", caption.get("captionLength", "medium")
        )
        caption["strictFocus"] = "strict_focus" in form
        caption["addQualityTags"] = "add_quality_tags" in form
        caption["shuffleTopics"] = "shuffle_topics" in form
        caption["processingDelay"] = float(
            form.get("processing_delay", caption.get("processingDelay", 1))
        )

        topics = []
        for i in range(6):
            topics.append(form.get(f"topic_{i + 1}", ""))
        STATE["settings"]["promptTopics"] = topics
        log("Configuration updated.")
    return templates.TemplateResponse(
        "partials/settings_panel.html",
        {
            "request": request,
            "settings": STATE["settings"],
            "selected_checkpoint": STATE["selected_checkpoint"],
            "selected_use_case": STATE["selected_use_case"],
            "backend": STATE["backend"],
            "gemini_api_key": STATE["gemini_api_key"],
            "grok_api_key": STATE["grok_api_key"],
            "grok_model": STATE["grok_model"],
            "allow_nsfw": STATE["allow_nsfw"],
            "simulate_failures": STATE["simulate_failures"],
            "tooltips": TOOLTIPS,
            "api_key_missing": api_key_missing(),
            "env_present": env_present,
            "checkpoint_order": CONFIG["ui_hints"]["checkpoint_order"],
            "use_case_order": CONFIG["ui_hints"]["use_case_order"],
            "include_console": True,
            "logs": STATE["logs"],
        },
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload_images(request: Request, files: list[UploadFile] = File(...)) -> HTMLResponse:
    added = 0
    for uploaded in files:
        if not uploaded.content_type or not uploaded.content_type.startswith("image/"):
            continue
        suffix = Path(uploaded.filename or "").suffix or ".jpg"
        stored_name = f"{uuid4().hex}{suffix}"
        target_path = UPLOAD_DIR / stored_name
        with target_path.open("wb") as target_file:
            target_file.write(await uploaded.read())
        STATE["images"].append(
            {
                "id": uuid4().hex,
                "index": len(STATE["images"]) + 1,
                "original_name": uploaded.filename or stored_name,
                "stored_name": stored_name,
                "preview_url": f"/uploads/{stored_name}",
                "status": "pending",
                "caption": "",
                "progress": 0,
            }
        )
        added += 1
    log(f"Loaded {added} new images.")
    save_metadata()
    return templates.TemplateResponse(
        "partials/workspace.html",
        {
            "request": request,
            "images": STATE["images"],
            "all_done": all_done(STATE["images"]),
            "can_process": can_process(STATE["images"]),
            "is_processing": STATE["is_processing"],
            "include_console": True,
            "logs": STATE["logs"],
        },
    )


@app.post("/process", response_class=HTMLResponse)
async def process_images(request: Request) -> HTMLResponse:
    if STATE["is_processing"] and any(
        image["status"] == "processing" for image in STATE["images"]
    ):
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )
    if STATE["is_processing"]:
        STATE["is_processing"] = False

    form = await request.form()
    if form:
        lora = STATE["settings"].setdefault("lora", {})
        lora["filePrefix"] = form.get("file_prefix", lora.get("filePrefix", ""))
        lora["keyword"] = form.get("keyword", lora.get("keyword", ""))

        caption = STATE["settings"].setdefault("captionControl", {})
        caption["guidance"] = form.get("caption_guidance", caption.get("guidance", ""))
        caption["guidanceStrength"] = float(
            form.get("guidance_strength", caption.get("guidanceStrength", 0.7))
        )
        caption["negativeHints"] = form.get(
            "negative_hints", caption.get("negativeHints", "")
        )
        caption["captionLength"] = form.get(
            "caption_length", caption.get("captionLength", "medium")
        )
        caption["subjectToken"] = form.get("subject_token", caption.get("subjectToken", ""))
        caption["addQualityTags"] = "add_quality_tags" in form
        caption["shuffleTopics"] = "shuffle_topics" in form
        caption["strictFocus"] = "strict_focus" in form
        caption["processingDelay"] = float(
            form.get("processing_delay", caption.get("processingDelay", 1))
        )

        topics = []
        for i in range(6):
            topics.append(form.get(f"topic_{i + 1}", ""))
        STATE["settings"]["promptTopics"] = topics
        print(
            "[process form] "
            f"file_prefix={lora.get('filePrefix')!r} "
            f"keyword={lora.get('keyword')!r} "
            f"subject_token={caption.get('subjectToken')!r} "
            f"guidance={caption.get('guidance')!r} "
            f"guidance_strength={caption.get('guidanceStrength')!r} "
            f"negative_hints={caption.get('negativeHints')!r} "
            f"caption_length={caption.get('captionLength')!r} "
            f"add_quality_tags={caption.get('addQualityTags')!r} "
            f"shuffle_topics={caption.get('shuffleTopics')!r} "
            f"strict_focus={caption.get('strictFocus')!r} "
            f"processing_delay={caption.get('processingDelay')!r} "
            f"topics={STATE['settings'].get('promptTopics')!r}"
        )

    for image in STATE["images"]:
        if image.get("status") == "processing":
            image["status"] = "pending"
            image["progress"] = 0

    if not STATE["images"]:
        log("No images to process.")
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    api_key = resolve_active_api_key()
    if not api_key:
        backend_label = "Grok (xAI)" if STATE["backend"] == "grok" else "Gemini"
        log(f"Missing API key for {backend_label}. Add it in Configuration → Captioning Backend.")
        STATE["is_processing"] = False
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    prefix = (STATE["settings"].get("lora", {}).get("filePrefix") or "").strip()
    if not prefix:
        prefix = "image"
    STATE["file_prefix"] = prefix

    queue = [
        image["id"]
        for image in STATE["images"]
        if image["status"] in {"pending", "error"}
    ]
    STATE["processing_queue"] = queue
    STATE["processing_index"] = 0
    STATE["processing_delay_remaining"] = 0.0
    STATE["is_processing"] = True
    log(f"Starting caption generation for {len(queue)} images...")
    master_prompt = build_ai_prompt(
        STATE["settings"],
        STATE["backend"],
        STATE["selected_checkpoint"],
        STATE["allow_nsfw"],
    )
    STATE["master_prompt"] = master_prompt
    print(f"[prompt] {master_prompt}")

    if queue:
        first_index = next_pending_image_index(queue, 0, STATE["images"])
        STATE["processing_index"] = first_index
    return templates.TemplateResponse(
        "partials/workspace.html",
        {
            "request": request,
            "images": STATE["images"],
            "all_done": all_done(STATE["images"]),
            "can_process": can_process(STATE["images"]),
            "is_processing": STATE["is_processing"],
            "include_console": True,
            "logs": STATE["logs"],
        },
    )


@app.get("/download", response_class=StreamingResponse)
def download_zip() -> StreamingResponse:
    if not all_done(STATE["images"]):
        log("Download blocked: processing incomplete.")
    prefix = STATE["settings"].get("lora", {}).get("filePrefix") or "image"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        done_images = [img for img in STATE["images"] if img["status"] == "done"]
        for idx, image in enumerate(done_images):
            extension = Path(image["original_name"]).suffix or ".jpg"
            image_name = f"{prefix}_{idx + 1:04d}{extension}"
            caption_name = f"{prefix}_{idx + 1:04d}.txt"
            image_path = UPLOAD_DIR / image["stored_name"]
            if image_path.exists():
                zip_file.write(image_path, arcname=image_name)
            caption_txt = image.get("caption_txt")
            if caption_txt:
                caption_path = UPLOAD_DIR / caption_txt
                if caption_path.exists():
                    zip_file.write(caption_path, arcname=caption_name)
                    continue
            zip_file.writestr(caption_name, image.get("caption", ""))
    buffer.seek(0)

    headers = {"Content-Disposition": f"attachment; filename={prefix}_dataset.zip"}
    return StreamingResponse(buffer, media_type="application/zip", headers=headers)


@app.post("/reset", response_class=HTMLResponse)
def reset_set(request: Request) -> HTMLResponse:
    STATE["is_processing"] = False
    STATE["processing_queue"] = []
    STATE["processing_index"] = 0
    STATE["processing_delay_remaining"] = 0.0
    for path in UPLOAD_DIR.iterdir():
        if not path.is_file():
            continue
        try:
            path.unlink()
        except OSError as exc:
            log(f"Failed to remove {path.name}: {exc}")
    STATE["images"].clear()
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
    log("Cleared current set. Ready for new uploads.")
    return templates.TemplateResponse(
        "partials/workspace.html",
        {
            "request": request,
            "images": STATE["images"],
            "all_done": all_done(STATE["images"]),
            "can_process": can_process(STATE["images"]),
            "is_processing": STATE["is_processing"],
            "include_console": True,
            "logs": STATE["logs"],
        },
    )


@app.get("/process/tick", response_class=HTMLResponse)
def process_tick(request: Request) -> HTMLResponse:
    if not STATE["is_processing"]:
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    queue = STATE["processing_queue"]
    if not queue:
        STATE["is_processing"] = False
        log("All images processed.")
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    if STATE["processing_delay_remaining"] > 0:
        STATE["processing_delay_remaining"] = max(
            0.0, STATE["processing_delay_remaining"] - 1.0
        )
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    current_index = STATE["processing_index"]
    if current_index >= len(queue):
        STATE["is_processing"] = False
        log("All images processed.")
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    current_id = queue[current_index]
    current_image = next(
        (image for image in STATE["images"] if image["id"] == current_id), None
    )
    if not current_image:
        STATE["processing_index"] += 1
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    if current_image["status"] == "done":
        STATE["processing_index"] += 1
        return templates.TemplateResponse(
            "partials/workspace.html",
            {
                "request": request,
                "images": STATE["images"],
                "all_done": all_done(STATE["images"]),
                "can_process": can_process(STATE["images"]),
                "is_processing": STATE["is_processing"],
                "include_console": True,
                "logs": STATE["logs"],
            },
        )

    if current_image["status"] != "processing":
        current_image["status"] = "processing"
        current_image["progress"] = 30
    else:
        if should_fail_image(current_image):
            error_message = "Error: simulated caption failure."
            current_image["status"] = "error"
            current_image["progress"] = 0
            current_image["caption"] = error_message
            log(f"Error processing {current_image['original_name']}: simulated failure")
        else:
            try:
                image_path = str(UPLOAD_DIR / current_image["stored_name"])
                if STATE["backend"] == "grok":
                    guidance_strength = (
                        STATE.get("settings", {})
                        .get("captionControl", {})
                        .get("guidanceStrength", 1.0)
                    )
                    raw_description = generate_grok_caption(
                        image_path=image_path,
                        prompt=STATE["master_prompt"],
                        api_key=resolve_active_api_key(),
                        model=STATE["grok_model"],
                        temperature=float(guidance_strength),
                    )
                else:
                    guidance_strength = (
                        STATE.get("settings", {})
                        .get("captionControl", {})
                        .get("guidanceStrength", 1.0)
                    )
                    raw_description = generate_gemini_caption(
                        image_path=image_path,
                        prompt=STATE["master_prompt"],
                        api_key=resolve_active_api_key(),
                        temperature=float(guidance_strength),
                    )
                print(f"[raw_caption] {raw_description}")
                final_caption = build_final_caption(
                    raw_description,
                    STATE["settings"],
                    STATE["selected_checkpoint"],
                    current_image.get("id"),
                )
                txt_prefix = STATE.get("file_prefix") or "image"
                txt_name = f"{txt_prefix}_{current_image['index']:04d}.txt"
                txt_path = UPLOAD_DIR / txt_name
                txt_path.write_text(final_caption)
                current_image["caption_txt"] = txt_name
                current_image["progress"] = 100
                current_image["status"] = "done"
                current_image["caption"] = final_caption
                log(f"Captioned {current_image['original_name']}.")
                print(f"[final_caption] {final_caption}")
            except Exception as exc:
                current_image["status"] = "error"
                current_image["progress"] = 0
                current_image["caption"] = f"Error: {exc}"
                log(f"Error processing {current_image['original_name']}: {exc}")
        save_metadata()
        STATE["processing_index"] += 1
        delay = (
            STATE["settings"]
            .get("captionControl", {})
            .get("processingDelay", 1)
        )
        STATE["processing_delay_remaining"] = max(0.0, float(delay))

    if all_done(STATE["images"]):
        STATE["is_processing"] = False
        log("All images processed.")

    return templates.TemplateResponse(
        "partials/workspace.html",
        {
            "request": request,
            "images": STATE["images"],
            "all_done": all_done(STATE["images"]),
            "can_process": can_process(STATE["images"]),
            "is_processing": STATE["is_processing"],
            "include_console": True,
            "logs": STATE["logs"],
        },
    )
