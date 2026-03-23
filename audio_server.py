"""
GCP Vertex AI Tools MCP Server
- Audio: Upload + transcribe with Gemini 2.5 Pro via Vertex AI
- Images: Generate with Gemini 2.5 Flash Image (Nano) — fastest/cheapest model
  Saves everything to gs://open-files-app organised by date.

Image generation models (from https://ai.google.dev/gemini-api/docs/image-generation):
  gemini-2.5-flash-image       — Nano (speed + efficiency, default)
  gemini-2.5-pro-image-preview — Pro (highest quality)
"""

import os
import json
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path

import google.auth
import google.auth.transport.requests
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("gcp-vertex-ai-tools")

PROJECT  = os.environ.get("GCP_PROJECT",   "ayusha-you2")
LOCATION = os.environ.get("GCP_LOCATION",  "us-central1")
BUCKET   = os.environ.get("AUDIO_BUCKET",  "open-files-app")
TRANSCRIBE_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro-preview-03-25")

# Nano image model (fast, efficient) — user-overridable
IMAGE_MODEL_NANO = os.environ.get("IMAGE_MODEL_NANO", "gemini-3.1-flash-image-preview")
IMAGE_MODEL_PRO  = os.environ.get("IMAGE_MODEL_PRO",  "gemini-3-pro-image-preview")

# Initialise Vertex AI (used for audio transcription)
vertexai.init(project=PROJECT, location=LOCATION)

# Initialise google-genai client pointed at Vertex AI (used for image generation)
_genai = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)

AUDIO_MIME_TYPES = {
    ".m4a": "audio/mp4", ".mp3": "audio/mpeg", ".wav": "audio/wav",
    ".ogg": "audio/ogg", ".flac": "audio/flac", ".aac": "audio/aac",
    ".webm": "audio/webm", ".opus": "audio/opus",
    ".mp4": "video/mp4",  ".mov": "video/quicktime",
}

VALID_ASPECT_RATIOS = {"1:1", "9:16", "16:9", "3:4", "4:3", "4:5", "5:4",
                       "21:9", "1:4", "4:1", "1:8", "8:1", "2:3", "3:2"}
VALID_IMAGE_SIZES   = {"512", "1K", "2K", "4K"}

# ---------------------------------------------------------------------------
# PROMPT TIPS (from official docs)
# ---------------------------------------------------------------------------
PROMPT_TIPS = {
    "general": [
        "Describe the scene narratively — don't just list keywords.",
        "Be specific about style, mood, lighting, and composition.",
        "Mention what you DON'T want by describing the opposite (no negative_prompt param — describe away from it).",
        "Shorter, cleaner prompts often outperform long lists of adjectives.",
    ],
    "photorealistic": [
        "Reference camera specs: 'shot on 35mm f/1.8, shallow depth of field'.",
        "Specify lighting: 'golden hour', 'studio three-point lighting', 'overcast diffuse light'.",
        "Add fine detail cues: 'sharp focus', '8K detail', 'photojournalistic'.",
    ],
    "stylized_assets": [
        "Explicitly state the art style: 'flat vector illustration', 'watercolour', 'pixel art'.",
        "Specify outline weight and background: 'thick black outline, white background'.",
        "For icons/stickers: 'sticker style, die-cut, no background'.",
    ],
    "text_in_image": [
        "Wrap text in quotes inside the prompt: 'a sign reading \"Hello World\"'.",
        "Specify font style and colour: 'bold sans-serif white text'.",
        "Keep text short — Gemini handles short words/phrases best.",
    ],
    "product_photography": [
        "Describe studio setup: 'white cyclorama, softbox lighting, 45-degree angle'.",
        "Mention surface material: 'marble surface', 'wooden table', 'reflective acrylic'.",
        "Add context: 'lifestyle shot with blurred kitchen background'.",
    ],
    "thinking_level": {
        "minimal": "Fastest, good for simple prompts.",
        "high":    "Best quality for complex multi-object scenes — slower.",
    },
    "aspect_ratios": list(VALID_ASPECT_RATIOS),
    "image_sizes":   list(VALID_IMAGE_SIZES),
    "models": {
        "nano (default)": IMAGE_MODEL_NANO + " — fastest, most cost-efficient",
        "pro":            IMAGE_MODEL_PRO  + " — highest quality, slower",
    },
}


def _ok(data) -> str:
    return json.dumps(data, default=str)

def _err(e: Exception) -> str:
    return json.dumps({"error": type(e).__name__, "message": str(e)})

def _storage_client():
    return storage.Client(project=PROJECT)

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def _signed_url(blob: storage.Blob, expiry_hours: int = 48) -> str:
    credentials, _ = google.auth.default()
    credentials.refresh(google.auth.transport.requests.Request())
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=expiry_hours),
        method="GET",
        service_account_email=credentials.service_account_email,
        access_token=credentials.token,
    )

def _audio_mime(filename: str) -> str:
    return AUDIO_MIME_TYPES.get(Path(filename).suffix.lower(), "audio/mp4")

def _save_and_sign(bucket_obj, gcs_path: str, data: bytes,
                   content_type: str, expiry_hours: int):
    blob = bucket_obj.blob(gcs_path)
    blob.upload_from_string(data, content_type=content_type)
    return blob, _signed_url(blob, expiry_hours)


# ===========================================================================
# SECTION 1 — AUDIO TRANSCRIPTION
# ===========================================================================

@mcp.tool()
def upload_and_transcribe(
    file_path: str = "",
    file_content_base64: str = "",
    filename: str = "",
    language: str = "en",
    date: str = "",
    expiry_hours: int = 48,
) -> str:
    """Upload an audio file to GCS and transcribe it with Gemini 2.5 Pro.

    Provide either file_path (local path) or file_content_base64 + filename.
    Stored at gs://open-files-app/recordings/YYYY-MM-DD/<filename>
    Transcript saved as <filename_no_ext>_transcript.txt in the same folder.

    Args:
        file_path: Absolute local path (m4a, mp3, wav, flac, ogg, aac)
        file_content_base64: Base64-encoded audio content (alternative)
        filename: Required when using file_content_base64
        language: BCP-47 language code (default: en)
        date: YYYY-MM-DD folder (default: today UTC)
        expiry_hours: Signed URL validity hours (default: 48)
    """
    try:
        recording_date = date or _today()
        if file_path:
            p = Path(file_path)
            if not p.exists():
                return _err(FileNotFoundError(f"Not found: {file_path}"))
            audio_bytes, fname = p.read_bytes(), filename or p.name
        elif file_content_base64:
            if not filename:
                return json.dumps({"error": "filename required with file_content_base64"})
            audio_bytes, fname = base64.b64decode(file_content_base64), filename
        else:
            return json.dumps({"error": "Provide file_path or file_content_base64+filename"})

        stem = Path(fname).stem
        audio_gcs = f"recordings/{recording_date}/{fname}"
        txt_gcs   = f"recordings/{recording_date}/{stem}_transcript.txt"

        client = _storage_client()
        bkt    = client.bucket(BUCKET)

        audio_blob, audio_url = _save_and_sign(bkt, audio_gcs, audio_bytes,
                                               _audio_mime(fname), expiry_hours)
        gcs_uri    = f"gs://{BUCKET}/{audio_gcs}"
        model      = GenerativeModel(TRANSCRIBE_MODEL)
        audio_part = Part.from_uri(gcs_uri, mime_type=_audio_mime(fname))
        transcript = model.generate_content([
            f"Transcribe this audio completely and accurately. Language: {language}. "
            "Include speaker labels if multiple speakers. Preserve paragraph breaks. "
            "Output transcript only.",
            audio_part,
        ]).text

        txt_blob, txt_url = _save_and_sign(bkt, txt_gcs,
                                           transcript.encode("utf-8"),
                                           "text/plain; charset=utf-8", expiry_hours)
        return _ok({
            "status": "success", "date": recording_date,
            "audioFile": {"gcsPath": gcs_uri, "signedUrl": audio_url, "sizeBytes": len(audio_bytes)},
            "transcript": {"text": transcript,
                           "gcsPath": f"gs://{BUCKET}/{txt_gcs}",
                           "signedUrl": txt_url},
        })
    except Exception as e:
        return _err(e)


@mcp.tool()
def transcribe_from_gcs(
    gcs_path: str,
    language: str = "en",
    expiry_hours: int = 48,
    overwrite: bool = False,
) -> str:
    """Transcribe an audio file already in gs://open-files-app.

    Args:
        gcs_path: GCS path or full gs:// URI
        language: Language code (default: en)
        expiry_hours: Signed URL validity hours (default: 48)
        overwrite: Re-transcribe even if cached transcript exists (default: False)
    """
    try:
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path.replace(f"gs://{BUCKET}/", "")
        fname        = Path(gcs_path).name
        date_folder  = "/".join(gcs_path.split("/")[:-1])
        txt_gcs      = f"{date_folder}/{Path(fname).stem}_transcript.txt"

        client   = _storage_client()
        bkt      = client.bucket(BUCKET)
        txt_blob = bkt.blob(txt_gcs)

        if txt_blob.exists() and not overwrite:
            transcript = txt_blob.download_as_text()
            return _ok({"status": "cached", "transcript": {
                "text": transcript,
                "gcsPath": f"gs://{BUCKET}/{txt_gcs}",
                "signedUrl": _signed_url(txt_blob, expiry_hours),
            }})

        gcs_uri    = f"gs://{BUCKET}/{gcs_path}"
        model      = GenerativeModel(TRANSCRIBE_MODEL)
        audio_part = Part.from_uri(gcs_uri, mime_type=_audio_mime(fname))
        transcript = model.generate_content([
            f"Transcribe this audio completely. Language: {language}. "
            "Speaker labels if multiple speakers. Output transcript only.",
            audio_part,
        ]).text

        _, txt_url = _save_and_sign(bkt, txt_gcs, transcript.encode("utf-8"),
                                    "text/plain; charset=utf-8", expiry_hours)
        return _ok({"status": "success", "transcript": {
            "text": transcript,
            "gcsPath": f"gs://{BUCKET}/{txt_gcs}",
            "signedUrl": txt_url,
        }})
    except Exception as e:
        return _err(e)


@mcp.tool()
def list_recordings(date: str = "") -> str:
    """List audio recordings and transcripts by date.

    Args:
        date: YYYY-MM-DD (default: today). Use 'all' for all dates.
    """
    try:
        prefix = "recordings/" if not date or date == "all" else f"recordings/{date}/"
        recordings: dict = {}
        for blob in _storage_client().list_blobs(BUCKET, prefix=prefix):
            parts = blob.name.split("/")
            if len(parts) < 3:
                continue
            d, fname = parts[1], parts[2]
            recordings.setdefault(d, {"audio": [], "transcripts": []})
            if fname.endswith("_transcript.txt"):
                recordings[d]["transcripts"].append(fname)
            else:
                recordings[d]["audio"].append({"name": fname, "sizeBytes": blob.size, "updated": blob.updated})
        return _ok({"recordings": recordings, "dates": sorted(recordings.keys(), reverse=True)})
    except Exception as e:
        return _err(e)


# ===========================================================================
# SECTION 2 — IMAGE GENERATION (Gemini Nano / Pro via Vertex AI)
# ===========================================================================

@mcp.tool()
def generate_image(
    prompt: str,
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    thinking_level: str = "minimal",
    model: str = "nano",
    date: str = "",
    filename: str = "",
    expiry_hours: int = 48,
    include_text: bool = False,
) -> str:
    """Generate an image with Gemini 2.5 Flash Image (Nano) or Pro via Vertex AI.
    Saves PNG to gs://open-files-app/generated/YYYY-MM-DD/ and returns signed URL + base64.

    Args:
        prompt: Image description (narrative style recommended — see get_prompt_tips)
        aspect_ratio: One of 1:1 16:9 9:16 4:3 3:4 4:5 5:4 21:9 2:3 3:2 1:4 4:1 1:8 8:1
        image_size: 512 | 1K | 2K | 4K (512 only on nano)
        thinking_level: minimal (fast) | high (best quality for complex scenes)
        model: nano (default, fastest) | pro (highest quality)
        date: YYYY-MM-DD folder (default: today UTC)
        filename: Output filename without extension (default: auto-generated)
        expiry_hours: Signed URL validity hours (default: 48)
        include_text: Also return a text description alongside the image (default: False)
    """
    try:
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            return json.dumps({"error": f"Invalid aspect_ratio. Choose from: {sorted(VALID_ASPECT_RATIOS)}"})
        if image_size not in VALID_IMAGE_SIZES:
            return json.dumps({"error": f"Invalid image_size. Choose from: {VALID_IMAGE_SIZES}"})

        model_id = IMAGE_MODEL_NANO if model == "nano" else IMAGE_MODEL_PRO
        gen_date = date or _today()
        ts       = datetime.now(timezone.utc).strftime("%H%M%S")
        fname    = (filename or f"img_{ts}") + ".png"
        gcs_path = f"generated/{gen_date}/{fname}"

        modalities = ["IMAGE", "TEXT"] if include_text else ["IMAGE"]

        response = _genai.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=modalities,
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                ),
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1 if thinking_level == "high" else 0,
                ),
            ),
        )

        image_data = None
        text_out   = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data  # bytes
            elif hasattr(part, "text") and part.text:
                text_out = part.text

        if not image_data:
            return json.dumps({"error": "No image returned by model", "response": str(response)})

        client   = _storage_client()
        bkt      = client.bucket(BUCKET)
        img_blob, signed = _save_and_sign(bkt, gcs_path, image_data, "image/png", expiry_hours)

        result = {
            "status": "success",
            "model": model_id,
            "prompt": prompt,
            "image": {
                "gcsPath":    f"gs://{BUCKET}/{gcs_path}",
                "signedUrl":  signed,
                "base64":     base64.b64encode(image_data).decode(),
                "aspectRatio": aspect_ratio,
                "imageSize":  image_size,
            },
        }
        if text_out:
            result["description"] = text_out
        return _ok(result)
    except Exception as e:
        return _err(e)


@mcp.tool()
def edit_image(
    prompt: str,
    image_gcs_path: str = "",
    image_base64: str = "",
    image_mime: str = "image/png",
    aspect_ratio: str = "1:1",
    image_size: str = "1K",
    thinking_level: str = "high",
    model: str = "nano",
    date: str = "",
    filename: str = "",
    expiry_hours: int = 48,
) -> str:
    """Edit or transform an existing image with a text prompt (image-to-image).
    Provide either image_gcs_path (file in GCS) or image_base64.

    Args:
        prompt: Editing instruction (e.g. 'make the background sunset', 'add snow')
        image_gcs_path: GCS path or gs:// URI of source image
        image_base64: Base64-encoded source image (alternative)
        image_mime: MIME type of source image (default: image/png)
        aspect_ratio: Output aspect ratio
        image_size: Output size: 512 | 1K | 2K | 4K
        thinking_level: minimal | high (default: high for editing)
        model: nano | pro
        date: YYYY-MM-DD folder (default: today UTC)
        filename: Output filename without extension
        expiry_hours: Signed URL validity hours
    """
    try:
        model_id = IMAGE_MODEL_NANO if model == "nano" else IMAGE_MODEL_PRO
        gen_date = date or _today()
        ts       = datetime.now(timezone.utc).strftime("%H%M%S")
        fname    = (filename or f"edited_{ts}") + ".png"
        gcs_path = f"generated/{gen_date}/{fname}"

        # Load source image
        if image_gcs_path:
            src = image_gcs_path.replace(f"gs://{BUCKET}/", "") if image_gcs_path.startswith("gs://") else image_gcs_path
            img_bytes = _storage_client().bucket(BUCKET).blob(src).download_as_bytes()
        elif image_base64:
            img_bytes = base64.b64decode(image_base64)
        else:
            return json.dumps({"error": "Provide image_gcs_path or image_base64"})

        img_part = types.Part.from_bytes(data=img_bytes, mime_type=image_mime)

        response = _genai.models.generate_content(
            model=model_id,
            contents=[prompt, img_part],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                ),
                thinking_config=types.ThinkingConfig(
                    thinking_budget=-1 if thinking_level == "high" else 0,
                ),
            ),
        )

        image_data, text_out = None, None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_data = part.inline_data.data
            elif hasattr(part, "text") and part.text:
                text_out = part.text

        if not image_data:
            return json.dumps({"error": "No image returned", "response": str(response)})

        _, signed = _save_and_sign(_storage_client().bucket(BUCKET), gcs_path,
                                   image_data, "image/png", expiry_hours)
        result = {
            "status": "success", "model": model_id, "prompt": prompt,
            "image": {
                "gcsPath": f"gs://{BUCKET}/{gcs_path}",
                "signedUrl": signed,
                "base64": base64.b64encode(image_data).decode(),
            },
        }
        if text_out:
            result["description"] = text_out
        return _ok(result)
    except Exception as e:
        return _err(e)


@mcp.tool()
def enhance_prompt(
    prompt: str,
    style: str = "photorealistic",
) -> str:
    """Use Gemini to enhance/expand a short image prompt into a detailed generation prompt.
    Run this BEFORE generate_image for best results.

    Args:
        prompt: Your short initial idea (e.g. 'a doctor consulting a patient')
        style: photorealistic | illustration | product | minimalist | cinematic
    """
    try:
        style_hints = {
            "photorealistic": "camera specs, lens, lighting setup, time of day, fine detail",
            "illustration":   "art style, colour palette, line weight, mood, background treatment",
            "product":        "studio lighting, surface material, angle, brand context, clean background",
            "minimalist":     "negative space, single subject, muted palette, simple composition",
            "cinematic":      "film grain, anamorphic lens, colour grade, dramatic lighting, aspect ratio",
        }
        hint = style_hints.get(style, style_hints["photorealistic"])
        model    = GenerativeModel(TRANSCRIBE_MODEL)
        enhanced = model.generate_content(
            f"You are an expert AI image prompt engineer. "
            f"Rewrite this prompt into a detailed, vivid {style} image generation prompt. "
            f"Include: {hint}. Keep it under 200 words. Be specific and concrete.\n\n"
            f"Original prompt: {prompt}\n\nEnhanced prompt:"
        ).text.strip()
        return _ok({"original": prompt, "enhanced": enhanced, "style": style})
    except Exception as e:
        return _err(e)


@mcp.tool()
def get_prompt_tips(category: str = "all") -> str:
    """Return image prompt tips and best practices from the Gemini image generation docs.

    Args:
        category: all | general | photorealistic | stylized_assets | text_in_image |
                  product_photography | thinking_level | aspect_ratios | models
    """
    if category == "all":
        return _ok(PROMPT_TIPS)
    if category in PROMPT_TIPS:
        return _ok({category: PROMPT_TIPS[category]})
    return json.dumps({"error": f"Unknown category. Choose from: {list(PROMPT_TIPS.keys())}"})


@mcp.tool()
def list_generated_images(date: str = "") -> str:
    """List generated images in gs://open-files-app/generated/.

    Args:
        date: YYYY-MM-DD (default: today). Use 'all' for all dates.
    """
    try:
        prefix = "generated/" if not date or date == "all" else f"generated/{date}/"
        images: dict = {}
        for blob in _storage_client().list_blobs(BUCKET, prefix=prefix):
            parts = blob.name.split("/")
            if len(parts) < 3:
                continue
            d = parts[1]
            images.setdefault(d, []).append({
                "name": parts[2], "gcsPath": f"gs://{BUCKET}/{blob.name}",
                "sizeBytes": blob.size, "updated": blob.updated,
            })
        return _ok({"images": images, "dates": sorted(images.keys(), reverse=True)})
    except Exception as e:
        return _err(e)


@mcp.tool()
def get_signed_url(gcs_path: str, expiry_hours: int = 48) -> str:
    """Get a signed URL for any file in gs://open-files-app.

    Args:
        gcs_path: GCS path or full gs:// URI
        expiry_hours: URL validity in hours (default: 48)
    """
    try:
        if gcs_path.startswith("gs://"):
            gcs_path = gcs_path.replace(f"gs://{BUCKET}/", "")
        client = _storage_client()
        blob   = client.bucket(BUCKET).blob(gcs_path)
        if not blob.exists():
            return json.dumps({"error": f"Not found: gs://{BUCKET}/{gcs_path}"})
        return _ok({"signedUrl": _signed_url(blob, expiry_hours),
                    "gcsPath": f"gs://{BUCKET}/{gcs_path}", "expiryHours": expiry_hours})
    except Exception as e:
        return _err(e)


# ===========================================================================
# ENTRYPOINT
# ===========================================================================

if __name__ == "__main__":
    port = os.environ.get("PORT")
    if port:
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(port)
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
