import base64
import io
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import magic
import pypdfium2 as pdfium
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from paddleocr import PaddleOCRVL
from PIL import Image


ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ALLOWED_IMAGE_MIMES = {
    "image/png", "image/jpeg", "image/bmp", "image/x-ms-bmp",
    "image/tiff", "image/webp",
}


DB_PATH = os.environ.get("DB_PATH", "/data/ocr.db")
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "/data/uploads")
DPI = int(os.environ.get("OCR_DPI", "200"))
API_KEY = os.environ.get("API_KEY", "")


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


IMAGE_DESCRIPTION_ENABLED = _env_bool("IMAGE_DESCRIPTION_ENABLED", False)
IMAGE_DESCRIPTION_PROVIDER = os.environ.get("IMAGE_DESCRIPTION_PROVIDER", "openai").lower()
IMAGE_DESCRIPTION_API_URL = os.environ.get("IMAGE_DESCRIPTION_API_URL", "https://api.openai.com/v1")
IMAGE_DESCRIPTION_API_KEY = os.environ.get("IMAGE_DESCRIPTION_API_KEY", "")
IMAGE_DESCRIPTION_API_VERSION = os.environ.get("IMAGE_DESCRIPTION_API_VERSION", "")
IMAGE_DESCRIPTION_API_MODE = os.environ.get("IMAGE_DESCRIPTION_API_MODE", "chat_completions").lower()
IMAGE_DESCRIPTION_MODEL = os.environ.get("IMAGE_DESCRIPTION_MODEL", "gpt-5.4")
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

_IMG_PATH_RE = re.compile(r"img_in_(?P<label>[a-z_]+?)_box_(\d+)_(\d+)_(\d+)_(\d+)")
_HTML_IMG_RE = re.compile(r'<img\s+[^>]*src="(?P<src>[^"]+)"[^>]*/?>', re.IGNORECASE)
_MD_IMG_RE = re.compile(r'!\[[^\]]*\]\((?P<src>[^)]+)\)')


_vision_client = None
_vision_client_lock = threading.Lock()


def _build_vision_client():
    from openai import AzureOpenAI, OpenAI

    if IMAGE_DESCRIPTION_PROVIDER == "azure":
        return AzureOpenAI(
            azure_endpoint=IMAGE_DESCRIPTION_API_URL,
            api_key=IMAGE_DESCRIPTION_API_KEY or "none",
            api_version=IMAGE_DESCRIPTION_API_VERSION,
            timeout=IMAGE_DESCRIPTION_TIMEOUT,
        )
    return OpenAI(
        base_url=IMAGE_DESCRIPTION_API_URL,
        api_key=IMAGE_DESCRIPTION_API_KEY or "none",
        timeout=IMAGE_DESCRIPTION_TIMEOUT,
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
    return IMAGE_DESCRIPTION_PROMPT_OVERRIDES.get(label.lower(), IMAGE_DESCRIPTION_DEFAULT_PROMPT)


def _encode_image(pil_image) -> str:
    img = pil_image
    if IMAGE_DESCRIPTION_MAX_EDGE_PX > 0 and max(img.size) > IMAGE_DESCRIPTION_MAX_EDGE_PX:
        img = img.copy()
        img.thumbnail((IMAGE_DESCRIPTION_MAX_EDGE_PX, IMAGE_DESCRIPTION_MAX_EDGE_PX))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _vision_call(client, data_url: str, prompt: str) -> str:
    if IMAGE_DESCRIPTION_API_MODE == "responses":
        resp = client.responses.create(
            model=IMAGE_DESCRIPTION_MODEL,
            input=[{"role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]}],
            timeout=IMAGE_DESCRIPTION_TIMEOUT,
        )
        return (resp.output_text or "").strip()
    resp = client.chat.completions.create(
        model=IMAGE_DESCRIPTION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        timeout=IMAGE_DESCRIPTION_TIMEOUT,
    )
    return (resp.choices[0].message.content or "").strip()


def _describe_one(client, pil_image, prompt: str) -> str:
    data_url = _encode_image(pil_image)
    last_err = None
    for attempt in range(IMAGE_DESCRIPTION_MAX_RETRIES + 1):
        try:
            return _vision_call(client, data_url, prompt)
        except Exception as e:
            last_err = e
            if attempt < IMAGE_DESCRIPTION_MAX_RETRIES:
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

        if label in NATIVE_RENDERED_LABELS:
            continue
        if label not in IMAGE_DESCRIPTION_LABELS:
            replacements[path] = ""
            continue

        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < IMAGE_DESCRIPTION_MIN_PIXELS:
            replacements[path] = ""
            continue

        if described >= IMAGE_DESCRIPTION_MAX_PER_PAGE:
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
            if IMAGE_DESCRIPTION_ON_ERROR == "fail":
                raise
            if IMAGE_DESCRIPTION_ON_ERROR == "placeholder":
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


def verify_api_key(request: Request):
    if not API_KEY:
        return
    if request.headers.get("X-API-Key") != API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


def init_db():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_pages INTEGER DEFAULT 0,
                processed_pages INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                page_num INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(job_id, page_num)
            );
        """)
        now = time.time()
        stale = db.execute("SELECT id FROM jobs WHERE status = 'processing'").fetchall()
        for row in stale:
            db.execute("DELETE FROM pages WHERE job_id = ?", (row["id"],))
        db.execute(
            "UPDATE jobs SET status = 'queued', processed_pages = 0, updated_at = ? WHERE status = 'processing'",
            (now,),
        )


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()



_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE = re.compile(r"<(th|td)[^>]*>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def _cell_text(raw: str) -> str:
    raw = re.sub(r"<br\s*/?>", " ", raw, flags=re.IGNORECASE)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = raw.replace("|", r"\|")
    return " ".join(raw.split())


def _table_to_markdown(inner_html: str) -> str:
    rows = []
    for tr in _TR_RE.finditer(inner_html):
        cells = [_cell_text(m.group(2)) for m in _CELL_RE.finditer(tr.group(1))]
        if cells and any(c for c in cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * width) + " |"]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def convert_html_tables(text: str) -> str:
    return _TABLE_RE.sub(lambda m: "\n\n" + _table_to_markdown(m.group(1)) + "\n\n", text)


def strip_html(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    return clean


def strip_image_tags(text: str) -> str:
    return re.sub(r"!\[.*?\]\(.*?\)\n*", "", text)



class OCRWorker:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._model = None
        self._cancelled: set[str] = set()
        self._cancel_lock = threading.Lock()

    def cancel_job(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.add(job_id)

    def _is_cancelled(self, job_id: str) -> bool:
        with self._cancel_lock:
            return job_id in self._cancelled

    def _clear_cancelled(self, job_id: str):
        with self._cancel_lock:
            self._cancelled.discard(job_id)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _load_model(self):
        if self._model is None:
            print("Loading PaddleOCR-VL model...")
            self._model = PaddleOCRVL()
            print("Model loaded.")
        return self._model

    def _run(self):
        ocr = self._load_model()
        while not self._stop.is_set():
            job = self._pick_next_job()
            if job is None:
                time.sleep(1)
                continue
            self._process_job(ocr, job)

    def _pick_next_job(self):
        with get_db() as db:
            row = db.execute(
                "SELECT id, filename FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row:
                now = time.time()
                db.execute(
                    "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ?",
                    (now, row["id"]),
                )
                return dict(row)
        return None

    def _process_job(self, ocr, job):
        job_id = job["id"]
        job_dir = Path(UPLOAD_DIR) / job_id
        input_candidates = list(job_dir.glob("input.*"))
        if not input_candidates:
            raise FileNotFoundError(f"no input file in {job_dir}")
        input_path = input_candidates[0]
        is_pdf = input_path.suffix.lower() == ".pdf"

        try:
            if is_pdf:
                pdf = pdfium.PdfDocument(str(input_path))
                total_pages = len(pdf)
            else:
                pdf = None
                total_pages = 1

            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET total_pages = ?, updated_at = ? WHERE id = ?",
                    (total_pages, time.time(), job_id),
                )

            scale = DPI / 72
            for page_idx in range(total_pages):
                if self._stop.is_set():
                    return
                if self._is_cancelled(job_id):
                    self._clear_cancelled(job_id)
                    with get_db() as db:
                        db.execute(
                            "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                            (time.time(), job_id),
                        )
                    print(f"[{job_id[:8]}] Job cancelled at page {page_idx + 1}/{total_pages}")
                    return

                if is_pdf:
                    page = pdf[page_idx]
                    bitmap = page.render(scale=scale)
                    pil_image = bitmap.to_pil()
                else:
                    pil_image = Image.open(input_path)
                    if pil_image.mode not in ("RGB", "L"):
                        pil_image = pil_image.convert("RGB")

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    pil_image.save(tmp.name)
                    tmp_path = tmp.name

                try:
                    result = ocr.predict(input=tmp_path)

                    markdown_parts = []
                    for res in result:
                        md_data = res._to_markdown(pretty=False)
                        if isinstance(md_data, dict):
                            text = md_data.get("markdown_texts") or md_data.get("markdown") or ""
                            images = md_data.get("markdown_images") or {}
                            if text:
                                if IMAGE_DESCRIPTION_ENABLED and images:
                                    text = describe_images(
                                        text, images,
                                        page_num=page_idx + 1, job_id=job_id,
                                    )
                                else:
                                    text = strip_image_tags(text)
                                markdown_parts.append(text)

                    if not markdown_parts:
                        markdown_parts = [self._extract_text(result)]

                    page_markdown = "\n\n".join(markdown_parts)
                    page_markdown = convert_html_tables(page_markdown)
                    page_markdown = strip_html(page_markdown)

                    now = time.time()
                    with get_db() as db:
                        db.execute(
                            "INSERT INTO pages (job_id, page_num, markdown, created_at) VALUES (?, ?, ?, ?)",
                            (job_id, page_idx + 1, page_markdown, now),
                        )
                        db.execute(
                            "UPDATE jobs SET processed_pages = ?, updated_at = ? WHERE id = ?",
                            (page_idx + 1, now, job_id),
                        )

                    print(f"[{job_id[:8]}] Page {page_idx + 1}/{total_pages} done")

                finally:
                    os.unlink(tmp_path)

            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET status = 'completed', updated_at = ? WHERE id = ?",
                    (time.time(), job_id),
                )
            print(f"[{job_id[:8]}] Job completed ({total_pages} pages)")

        except Exception as e:
            import traceback
            traceback.print_exc()
            with get_db() as db:
                db.execute(
                    "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
                    (str(e), time.time(), job_id),
                )

    def _extract_text(self, result):
        texts = []
        for res in result:
            if hasattr(res, "rec_text"):
                texts.append(res.rec_text)
            elif hasattr(res, "text"):
                texts.append(res.text)
            else:
                texts.append(str(res))
        return "\n".join(texts)


worker = OCRWorker()


app = FastAPI(title="PaddleOCR API", version="1.0.0", dependencies=[Depends(verify_api_key)])


@app.on_event("startup")
def startup():
    init_db()
    Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    worker.start()


@app.on_event("shutdown")
def shutdown():
    worker.stop()


@app.post("/ocr")
async def submit_job(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    is_pdf = suffix == ".pdf"
    is_image = suffix in ALLOWED_IMAGE_EXTS
    if not (is_pdf or is_image):
        raise HTTPException(
            400,
            "Only PDF and image files (PNG, JPG, JPEG, BMP, TIFF, WEBP) are supported",
        )

    content = await file.read()
    mime = magic.from_buffer(content, mime=True)
    if is_pdf and mime != "application/pdf":
        raise HTTPException(400, f"File is not a valid PDF (detected: {mime})")
    if is_image and mime not in ALLOWED_IMAGE_MIMES:
        raise HTTPException(400, f"File is not a valid image (detected: {mime})")

    job_id = uuid.uuid4().hex
    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(content)

    now = time.time()
    with get_db() as db:
        db.execute(
            "INSERT INTO jobs (id, filename, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)",
            (job_id, file.filename, now, now),
        )

    return {"job_id": job_id, "filename": file.filename, "status": "queued"}


@app.get("/ocr/{job_id}")
def get_job_status(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not job:
        raise HTTPException(404, "Job not found")

    return {
        "job_id": job["id"],
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "error": job["error"],
    }


@app.get("/ocr/{job_id}/pages/{page_num}")
def get_page(job_id: str, page_num: int):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        page = db.execute(
            "SELECT * FROM pages WHERE job_id = ? AND page_num = ?",
            (job_id, page_num),
        ).fetchone()

    if not page:
        if page_num > job["total_pages"] and job["total_pages"] > 0:
            raise HTTPException(404, f"Page {page_num} does not exist (total: {job['total_pages']})")
        raise HTTPException(202, f"Page {page_num} not yet processed")

    return {
        "job_id": job_id,
        "page_num": page["page_num"],
        "markdown": page["markdown"],
    }


@app.get("/ocr/{job_id}/result")
def get_full_result(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        pages = db.execute(
            "SELECT page_num, markdown FROM pages WHERE job_id = ? ORDER BY page_num",
            (job_id,),
        ).fetchall()

    return {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "pages": [{"page_num": p["page_num"], "markdown": p["markdown"]} for p in pages],
    }


@app.post("/ocr/{job_id}/cancel")
def cancel_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("queued", "processing"):
            raise HTTPException(400, f"Job cannot be cancelled (status: {job['status']})")
        if job["status"] == "queued":
            db.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?",
                (time.time(), job_id),
            )
        else:
            worker.cancel_job(job_id)
    return {"job_id": job_id, "status": "cancelling" if job["status"] == "processing" else "cancelled"}


@app.delete("/ocr/{job_id}")
def delete_job(job_id: str):
    with get_db() as db:
        job = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        db.execute("DELETE FROM pages WHERE job_id = ?", (job_id,))
        db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    job_dir = Path(UPLOAD_DIR) / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)

    return {"status": "deleted"}


@app.get("/jobs")
def list_jobs():
    with get_db() as db:
        jobs = db.execute(
            "SELECT id, filename, status, total_pages, processed_pages, created_at FROM jobs ORDER BY created_at DESC"
        ).fetchall()

    return {
        "jobs": [
            {
                "job_id": j["id"],
                "filename": j["filename"],
                "status": j["status"],
                "total_pages": j["total_pages"],
                "processed_pages": j["processed_pages"],
            }
            for j in jobs
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
