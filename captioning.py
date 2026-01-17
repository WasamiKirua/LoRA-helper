from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

import httpx
from google import genai
from google.genai import types
from openai import OpenAI
DEFAULT_PROMPT = (
    "A photorealistic portrait of a specific person, capturing their distinct facial "
    "features, hair, and expression."
)


def build_ai_prompt(
    settings: dict,
    backend: str,
    selected_checkpoint: str,
    allow_nsfw: bool,
) -> str:
    caption_control = (settings or {}).get("captionControl")
    if not caption_control:
        return DEFAULT_PROMPT

    prompt_parts = [caption_control.get("guidance") or DEFAULT_PROMPT]
    negative_hints = caption_control.get("negativeHints")
    if negative_hints:
        prompt_parts.append(f"\nAVOID describing elements like: {negative_hints}.")

    caption_length = caption_control.get("captionLength") or "medium"
    prompt_parts.append(f"\nFocus on creating a caption of {caption_length} length.")

    if backend == "grok" and allow_nsfw:
        prompt_parts.append(
            "\nDescribe NSFW or explicit content factually without censoring or refusing."
        )

    if selected_checkpoint == "Z-Image":
        prompt_parts.append(
            "\nRespond with 1-2 natural sentences, put the main subject first, and include "
            "camera distance/framing and lighting. Avoid bullet lists."
        )

    return " ".join(prompt_parts)


def guess_mime_type(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or "image/jpeg"


def generate_gemini_caption(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = "gemini-3-flash-preview",
    temperature: float = 1.0,
) -> str:
    image_bytes = Path(image_path).read_bytes()
    mime_type = guess_mime_type(image_path)

    print("=== Gemini Captioning ===")
    print(prompt)
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt,
            ],
            config=types.GenerateContentConfig(
            temperature=temperature
    )
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini response was empty.")
    return text.strip()


def generate_grok_caption(
    image_path: str,
    prompt: str,
    api_key: str,
    model: str = "grok-2-vision-latest",
    detail: str = "high",
    temperature: float = 1.0,
) -> str:
    image_bytes = Path(image_path).read_bytes()
    mime_type = guess_mime_type(image_path)
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{encoded}"

    
    print("=== Grok Captioning ===")


    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            timeout=httpx.Timeout(3600.0),
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": detail,
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ]
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            store=False,
        )
    except Exception as exc:
        error_body = ""
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if getattr(exc, "response", None) is not None:
            try:
                error_body = exc.response.text
            except Exception:
                error_body = ""
        status_line = f" {status_code}" if status_code else ""
        detail_line = f": {error_body}" if error_body else ""
        raise RuntimeError(f"Grok API HTTP{status_line}{detail_line} {exc}") from exc

    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", "") in {"output_text", "text"}:
                text = getattr(content, "text", "")
                if text:
                    return str(text).strip()

    raise RuntimeError("Grok response was empty.")


def build_final_caption(raw_description: str, settings: dict, checkpoint: str) -> str:
    lora = settings.get("lora", {}) if settings else {}
    caption_control = settings.get("captionControl", {}) if settings else {}
    prompt_topics = settings.get("promptTopics", []) if settings else []
    keyword = (lora.get("keyword") or "").strip()
    subject_token = (caption_control.get("subjectToken") or "").strip()

    model = checkpoint or lora.get("targetBaseModel") or "SDXL"
    cleaned_raw = (raw_description or "").strip()
    is_sd = model in {"SDXL", "SD-1.5"}
    is_flux = model in {"Flux", "Chroma"}
    is_wan = model == "WAN-2.2"
    is_qwen = model in {"Qwen-Image", "Qwen-Image-Edit"}
    is_z_image = model == "Z-Image"

    if is_flux or is_wan or is_qwen or is_z_image:
        text = cleaned_raw
        if is_qwen:
            text = " ".join(part for part in text.split() if not part.startswith("<"))
        if is_z_image:
            main_subject = " ".join(part for part in [keyword, subject_token] if part)
            base_sentence = f"{main_subject}." if main_subject else ""
            detail = " ".join(cleaned_raw.split())
            topics = [topic.strip() for topic in prompt_topics if topic and topic.strip()]
            topics_sentence = f"Style cues: {', '.join(topics)}." if topics else ""
            quality_line = (
                "High quality, photorealistic detail."
                if caption_control.get("addQualityTags")
                else ""
            )
            return " ".join(
                part for part in [base_sentence, detail, topics_sentence, quality_line] if part
            ).strip()

        trigger = " ".join(part for part in [keyword, subject_token] if part).strip()
        if (is_flux or is_qwen) and trigger and trigger.lower() not in text.lower():
            text = text.rstrip(".") + ". Featuring " + trigger + "."
        return text.strip()

    parts = []
    main_subject = " ".join(part for part in [keyword, subject_token] if part)
    if main_subject:
        parts.append(main_subject)

    cleaned_description = " ".join((raw_description or "").replace("\n", " ").split())
    if cleaned_description:
        parts.append(cleaned_description)

    topics = [topic.strip() for topic in prompt_topics if topic and topic.strip()]
    if topics:
        parts.extend(topics)

    if caption_control.get("addQualityTags"):
        parts.extend(["best quality", "high resolution"])

    final_caption = ", ".join(parts)
    if is_sd:
        tokens = [token.strip() for token in final_caption.split(",") if token.strip()]
        final_caption = ", ".join(tokens[:40])
    return final_caption.replace(", ,", ",").rstrip(",").strip()
