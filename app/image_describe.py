import base64
import io
import os
import re
import threading
import time

from app import config
from app.markdown import strip_image_tags


_IMG_PATH_RE = re.compile(r"img_in_(?P<label>[a-z_]+?)_box_(\d+)_(\d+)_(\d+)_(\d+)")
_HTML_IMG_RE = re.compile(r'<img\s+[^>]*src="(?P<src>[^"]+)"[^>]*/?>', re.IGNORECASE)
_MD_IMG_RE = re.compile(r'!\[[^\]]*\]\((?P<src>[^)]+)\)')


_vision_client = None
_vision_client_lock = threading.Lock()


def _build_vision_client():
    if config.IMAGE_DESCRIPTION_PROVIDER == "anthropic":
        from anthropic import Anthropic

        return Anthropic(
            api_key=config.IMAGE_DESCRIPTION_API_KEY or "none",
            timeout=config.IMAGE_DESCRIPTION_TIMEOUT,
        )

    from openai import AzureOpenAI, OpenAI

    if config.IMAGE_DESCRIPTION_PROVIDER == "azure":
        return AzureOpenAI(
            azure_endpoint=config.IMAGE_DESCRIPTION_API_URL,
            api_key=config.IMAGE_DESCRIPTION_API_KEY or "none",
            api_version=config.IMAGE_DESCRIPTION_API_VERSION,
            timeout=config.IMAGE_DESCRIPTION_TIMEOUT,
        )
    return OpenAI(
        base_url=config.IMAGE_DESCRIPTION_API_URL,
        api_key=config.IMAGE_DESCRIPTION_API_KEY or "none",
        timeout=config.IMAGE_DESCRIPTION_TIMEOUT,
    )


def _get_vision_client():
    global _vision_client
    if _vision_client is None:
        with _vision_client_lock:
            if _vision_client is None:
                _vision_client = _build_vision_client()
    return _vision_client


def _parse_image_path(path: str):
    name = os.path.basename(path)
    m = _IMG_PATH_RE.search(name)
    if not m:
        return None
    label = m.group("label").lower()
    x1, y1, x2, y2 = (int(m.group(i)) for i in (2, 3, 4, 5))
    return label, (x1, y1, x2, y2)


def _prompt_for(label: str) -> str:
    return config.IMAGE_DESCRIPTION_PROMPT_OVERRIDES.get(label.lower(), config.IMAGE_DESCRIPTION_DEFAULT_PROMPT)


def _encode_image(pil_image) -> str:
    img = pil_image
    if config.IMAGE_DESCRIPTION_MAX_EDGE_PX > 0 and max(img.size) > config.IMAGE_DESCRIPTION_MAX_EDGE_PX:
        img = img.copy()
        img.thumbnail((config.IMAGE_DESCRIPTION_MAX_EDGE_PX, config.IMAGE_DESCRIPTION_MAX_EDGE_PX))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _vision_call(client, data_url: str, prompt: str) -> str:
    if config.IMAGE_DESCRIPTION_PROVIDER == "anthropic":
        resp = client.messages.create(
            model=config.IMAGE_DESCRIPTION_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": data_url.split(",", 1)[1],
                }},
                {"type": "text", "text": prompt},
            ]}],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("model refused the request")
        return "".join(block.text for block in resp.content if block.type == "text").strip()
    if config.IMAGE_DESCRIPTION_API_MODE == "responses":
        resp = client.responses.create(
            model=config.IMAGE_DESCRIPTION_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]}],
            timeout=config.IMAGE_DESCRIPTION_TIMEOUT,
        )
        return (resp.output_text or "").strip()
    resp = client.chat.completions.create(
        model=config.IMAGE_DESCRIPTION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        timeout=config.IMAGE_DESCRIPTION_TIMEOUT,
    )
    return (resp.choices[0].message.content or "").strip()


def _describe_one(client, pil_image, prompt: str) -> str:
    data_url = _encode_image(pil_image)
    last_err = None
    for attempt in range(config.IMAGE_DESCRIPTION_MAX_RETRIES + 1):
        try:
            return _vision_call(client, data_url, prompt)
        except Exception as e:
            last_err = e
            if attempt < config.IMAGE_DESCRIPTION_MAX_RETRIES:
                time.sleep(min(2 ** attempt, 5))
    raise last_err


def _replace_image_tags(text: str, replacements: dict) -> str:
    def _sub(match):
        src = match.group("src")
        key = _match_replacement_key(src, replacements)
        if key is None:
            return ""
        return replacements[key]

    text = _HTML_IMG_RE.sub(_sub, text)
    text = _MD_IMG_RE.sub(_sub, text)
    text = re.sub(r'<div[^>]*>\s*</div>', "", text)
    text = re.sub(r'\n{3,}', "\n\n", text)
    return text


def _match_replacement_key(src: str, replacements: dict):
    if src in replacements:
        return src
    base = os.path.basename(src)
    for key in replacements:
        if os.path.basename(key) == base:
            return key
    return None


def describe_images(text: str, images: dict, page_num: int = 0, job_id: str = "") -> str:
    if not text or not images:
        return strip_image_tags(text)

    referenced = set()
    for m in _HTML_IMG_RE.finditer(text):
        referenced.add(m.group("src"))
    for m in _MD_IMG_RE.finditer(text):
        referenced.add(m.group("src"))

    client = None
    replacements: dict = {}
    described = 0

    for path, pil_image in images.items():
        if path not in referenced:
            base = os.path.basename(path)
            if not any(os.path.basename(r) == base for r in referenced):
                continue

        parsed = _parse_image_path(path)
        if parsed is None:
            replacements[path] = ""
            continue
        label, (x1, y1, x2, y2) = parsed

        if label in config.NATIVE_RENDERED_LABELS:
            continue
        if label not in config.IMAGE_DESCRIPTION_LABELS:
            replacements[path] = ""
            continue

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < config.IMAGE_DESCRIPTION_MIN_PIXELS:
            replacements[path] = ""
            continue

        if described >= config.IMAGE_DESCRIPTION_MAX_PER_PAGE:
            replacements[path] = ""
            continue

        if client is None:
            client = _get_vision_client()

        prompt = _prompt_for(label)
        label_display = label.replace("_", " ").title()

        try:
            desc = _describe_one(client, pil_image, prompt)
        except Exception as e:
            print(f"[image-desc] job={job_id[:8]} page={page_num} label={label} error: {e}")
            if config.IMAGE_DESCRIPTION_ON_ERROR == "fail":
                raise
            if config.IMAGE_DESCRIPTION_ON_ERROR == "placeholder":
                replacements[path] = f"> **[{label_display}]** [image description unavailable]"
            else:
                replacements[path] = ""
            continue

        if not desc:
            replacements[path] = ""
            continue

        replacements[path] = f"> **[{label_display}]** {desc}"
        described += 1

    return _replace_image_tags(text, replacements)
