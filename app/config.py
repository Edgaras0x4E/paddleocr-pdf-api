import os


RUN_MODE = os.environ.get("RUN_MODE", "server").strip().lower()
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
BACKEND = "postgres" if DATABASE_URL else "sqlite"


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp",
    "image/tiff", "image/webp",
}


DB_PATH = os.environ.get("DB_PATH", "/data/ocr.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/data/uploads")
DPI = int(os.environ.get("OCR_DPI", "200"))
API_KEY = os.environ.get("API_KEY", "")
PIPELINE_VERSION = os.environ.get("OCR_PIPELINE_VERSION", "v1.6")

_MARKDOWN_IGNORE_LABELS_RAW = os.environ.get("OCR_MARKDOWN_IGNORE_LABELS")
MARKDOWN_IGNORE_LABELS = (
    None
    if _MARKDOWN_IGNORE_LABELS_RAW is None
    else [lbl.strip().lower() for lbl in _MARKDOWN_IGNORE_LABELS_RAW.split(",") if lbl.strip()]
)

OCR_ENGINE = os.environ.get("OCR_ENGINE", "vl").strip().lower()
OCR_ENGINES = {"vl", "text", "structure"}
if OCR_ENGINE not in OCR_ENGINES:
    raise ValueError(f"Invalid OCR_ENGINE '{OCR_ENGINE}'. Valid values: {sorted(OCR_ENGINES)}")


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


BOUNDING_BOXES = _env_bool("OCR_BOUNDING_BOXES", False)

IMAGE_DESCRIPTION_ENABLED = _env_bool("IMAGE_DESCRIPTION_ENABLED", False)
IMAGE_DESCRIPTION_PROVIDER = os.environ.get("IMAGE_DESCRIPTION_PROVIDER", "openai").lower()
IMAGE_DESCRIPTION_API_URL = os.environ.get("IMAGE_DESCRIPTION_API_URL", "https://api.openai.com/v1")
IMAGE_DESCRIPTION_API_KEY = os.environ.get("IMAGE_DESCRIPTION_API_KEY", "")
IMAGE_DESCRIPTION_API_VERSION = os.environ.get("IMAGE_DESCRIPTION_API_VERSION", "")
IMAGE_DESCRIPTION_API_MODE = os.environ.get("IMAGE_DESCRIPTION_API_MODE", "chat_completions").lower()
IMAGE_DESCRIPTION_MODEL = os.environ.get("IMAGE_DESCRIPTION_MODEL", "gpt-5.5")
IMAGE_DESCRIPTION_DEFAULT_PROMPT = os.environ.get(
    "IMAGE_DESCRIPTION_PROMPT",
    "Describe this image from a document concisely. Focus on content relevant to "
    "understanding the document (what's shown, any text, data, or diagram meaning). "
    "Do not speculate.",
)
IMAGE_DESCRIPTION_LABELS = {
    lbl.strip().lower()
    for lbl in os.environ.get(
        "IMAGE_DESCRIPTION_LABELS",
        "image,chart,seal,header_image,footer_image",
    ).split(",")
    if lbl.strip()
}
IMAGE_DESCRIPTION_MIN_PIXELS = int(os.environ.get("IMAGE_DESCRIPTION_MIN_PIXELS", "10000"))
IMAGE_DESCRIPTION_MAX_EDGE_PX = int(os.environ.get("IMAGE_DESCRIPTION_MAX_EDGE_PX", "1568"))
IMAGE_DESCRIPTION_MAX_PER_PAGE = int(os.environ.get("IMAGE_DESCRIPTION_MAX_PER_PAGE", "10"))
IMAGE_DESCRIPTION_TIMEOUT = int(os.environ.get("IMAGE_DESCRIPTION_TIMEOUT", "60"))
IMAGE_DESCRIPTION_MAX_RETRIES = int(os.environ.get("IMAGE_DESCRIPTION_MAX_RETRIES", "2"))
IMAGE_DESCRIPTION_ON_ERROR = os.environ.get("IMAGE_DESCRIPTION_ON_ERROR", "skip").lower()

IMAGE_DESCRIPTION_PROMPT_OVERRIDES = {
    key[len("IMAGE_DESCRIPTION_PROMPT_"):].lower(): val
    for key, val in os.environ.items()
    if key.startswith("IMAGE_DESCRIPTION_PROMPT_") and val
}

NATIVE_RENDERED_LABELS = {"table", "formula"}
