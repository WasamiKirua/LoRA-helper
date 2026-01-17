# LoRA Helper

LoRA Helper is a local web app for generating consistent training captions for LoRA datasets. It lets you upload a batch of images, choose a target base model and training goal, tune caption controls, and download a ZIP with images + paired `.txt` captions formatted for LoRA training.

## Installation (uv)

Requirements:
- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

## Run

```bash
uv run uvicorn main:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

Optional `.env` (auto-updated when you save keys in the UI):

```bash
GEMINI_API_KEY=...
GROK_API_KEY=...
GROK_MODEL=grok-2-vision-latest
```

## UI walkthrough

1. Open the app and choose a Captioning Backend (Gemini or Grok).
2. Paste the API key for the selected backend.
3. Pick a Training Goal and Target Base Model (these load tuned defaults).
4. Adjust LoRA + Caption Control settings as needed.
5. Drop images into the workspace.
6. Click **Process Images** to generate captions.
7. Click **Download ZIP** to get the images and `.txt` captions.
8. Use **New Set** to clear the workspace for the next batch.

## Settings and how they affect prompts

### Captioning Backend
- **Service**: Chooses Gemini or Grok for caption generation. This determines which API is called.
- **API Key**: Required for the selected backend; saved into `.env` on the server.
- **Grok Vision Model**: Sets the Grok model name used for captioning.
- **Allow NSFW image captions** (Grok only): Adds an instruction telling Grok to describe NSFW content without censoring.
- **Simulate caption failures**: Forces periodic failures to test error handling; does not call APIs.

### LoRA Settings
- **Training Goal**: Loads a preset tuned for the goal (identity, likeness, cinematic, etc.). Changing this resets the settings to those defaults, so re-apply manual tweaks afterward.
- **File prefix**: Prefix for output filenames and caption files (e.g., `prefix_0001.jpg` + `prefix_0001.txt`).
- **Keyword**: Main trigger word for your subject; it is injected into the final captions.
- **Target Base Model**: Loads base-model-specific defaults and changes how captions are post-processed (e.g., SDXL vs Flux vs Qwen vs Z-Image formatting).

### Caption Control
- **Caption Guidance**: The primary instruction for the AI; this is the most influential prompt component.
- **Guidance Strength**: Passed as model temperature. Higher values generally increase variation and loosen adherence; lower values are more literal and stable.
- **Negative Hints**: Appends an instruction to avoid specific elements (e.g., “AVOID describing elements like: …”).
- **Caption Length**: Adds a target length instruction (short/medium/long) to the prompt.
- **Strict focus (subject-only)**: Adds an instruction to ignore background/secondary elements.
- **Add quality tags**: Appends `best quality` and `high resolution` to the final captions.
- **Shuffle topics per image**: Shuffles Prompt Topics using a stable per-image seed, changing the order without losing coverage.
- **Subject Token**: Secondary token (e.g., `<sks>`) appended with the keyword in the final caption.
- **Processing Delay (seconds)**: Wait time between image requests to reduce API rate-limit errors.

### Prompt Topics
- **Topic 1–6**: Keywords added to every caption to reinforce style or content. These are appended after the main description and keyword/subject token, and can be shuffled if enabled.

## What changes the prompt output

The prompt sent to the AI is built from:
- Caption Guidance (base sentence)
- Strict Focus (adds subject-only instruction)
- Negative Hints (adds avoid list)
- Caption Length (adds length target)
- Allow NSFW (adds explicit-content instruction for Grok)
- Target Base Model (adds model-specific instructions, e.g., Z-Image formatting)

The final saved caption is then post-processed using:
- Keyword + Subject Token (prepended)
- Prompt Topics (appended)
- Add Quality Tags (appended)
- Shuffle Topics (order changes per image)
- Target Base Model (formatting differences for SDXL/Flux/Qwen/WAN/Z-Image)

## Troubleshooting

- **Missing API key**: The app will block processing until you add a key in Configuration → Captioning Backend.
- **Rate-limited or slow responses**: Increase Processing Delay and retry.
- **Empty or odd captions**: Lower Guidance Strength or simplify Caption Guidance; some models are sensitive to overlong instructions.
