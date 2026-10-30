import json
import os
import shutil
import signal
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import magic
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app import config
from app import db
from app.worker import OCRWorker


def verify_api_key(request: Request):
    if not config.API_KEY:
        return
    if request.headers.get("X-API-Key") != config.API_KEY:
        raise HTTPException(401, "Invalid or missing API key")


worker = OCRWorker()


@asynccontextmanager
async def lifespan(app):
    db.init_db()
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    if config.BACKEND == "sqlite":
        db.recover_sqlite_jobs()
    worker.start()
    try:
        yield
    finally:
        worker.stop()
        worker.join()
        worker.close()


app = FastAPI(
    title="PaddleOCR API",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
    lifespan=lifespan,
)


def _store_job(content, suffix, is_pdf, is_image, filename):
    mime = magic.from_buffer(content, mime=True)
    if is_pdf and mime != "application/pdf":
        raise HTTPException(400, f"File is not a valid PDF (detected: {mime})")
    if is_image and mime not in config.ALLOWED_IMAGE_MIMES:
        raise HTTPException(400, f"File is not a valid image (detected: {mime})")

    job_id = uuid.uuid4().hex
    job_dir = Path(config.UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)

    input_path = job_dir / f"input{suffix}"
    input_path.write_bytes(content)

    now = time.time()
    with db.get_db() as conn:
        conn.execute(
            db._q("INSERT INTO jobs (id, filename, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)"),
            (job_id, filename, now, now),
        )

    return {"job_id": job_id, "filename": filename, "status": "queued"}


@app.post("/ocr")
async def submit_job(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    is_pdf = suffix == ".pdf"
    is_image = suffix in config.ALLOWED_IMAGE_EXTS
    if not (is_pdf or is_image):
        raise HTTPException(
            400,
            "Only PDF and image files (PNG, JPG, JPEG, BMP, TIFF, WEBP) are supported",
        )

    content = await file.read()
    return await run_in_threadpool(_store_job, content, suffix, is_pdf, is_image, file.filename)


@app.get("/ocr/{job_id}")
def get_job_status(job_id: str):
    with db.get_db() as conn:
        job = conn.execute(db._q("SELECT * FROM jobs WHERE id = ?"), (job_id,)).fetchone()
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


def _page_body(page):
    body = {"page_num": page["page_num"], "markdown": page["markdown"]}
    if config.BOUNDING_BOXES:
        body["bounding_boxes"] = json.loads(page["bounding_boxes"]) if page["bounding_boxes"] else None
    return body


@app.get("/ocr/{job_id}/pages/{page_num}")
def get_page(job_id: str, page_num: int):
    with db.get_db() as conn:
        job = conn.execute(db._q("SELECT * FROM jobs WHERE id = ?"), (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        page = conn.execute(
            db._q("SELECT * FROM pages WHERE job_id = ? AND page_num = ?"),
            (job_id, page_num),
        ).fetchone()

    if not page:
        if page_num > job["total_pages"] and job["total_pages"] > 0:
            raise HTTPException(404, f"Page {page_num} does not exist (total: {job['total_pages']})")
        raise HTTPException(202, f"Page {page_num} not yet processed")

    return {"job_id": job_id, **_page_body(page)}


@app.get("/ocr/{job_id}/result")
def get_full_result(job_id: str):
    with db.get_db() as conn:
        job = conn.execute(db._q("SELECT * FROM jobs WHERE id = ?"), (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")

        pages = conn.execute(
            db._q("SELECT page_num, markdown, bounding_boxes FROM pages WHERE job_id = ? ORDER BY page_num"),
            (job_id,),
        ).fetchall()

    return {
        "job_id": job_id,
        "filename": job["filename"],
        "status": job["status"],
        "total_pages": job["total_pages"],
        "processed_pages": job["processed_pages"],
        "pages": [_page_body(p) for p in pages],
    }


@app.post("/ocr/{job_id}/cancel")
def cancel_job(job_id: str):
    with db.get_db() as conn:
        job = conn.execute(db._q("SELECT * FROM jobs WHERE id = ?"), (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] not in ("queued", "processing"):
            raise HTTPException(400, f"Job cannot be cancelled (status: {job['status']})")
        if job["status"] == "queued":
            conn.execute(
                db._q("UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ?"),
                (time.time(), job_id),
            )
        else:
            conn.execute(
                db._q("UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?"),
                (time.time(), job_id),
            )
    return {"job_id": job_id, "status": "cancelling" if job["status"] == "processing" else "cancelled"}


@app.delete("/ocr/{job_id}")
def delete_job(job_id: str):
    with db.get_db() as conn:
        job = conn.execute(db._q("SELECT * FROM jobs WHERE id = ?"), (job_id,)).fetchone()
        if not job:
            raise HTTPException(404, "Job not found")
        conn.execute(db._q("DELETE FROM pages WHERE job_id = ?"), (job_id,))
        conn.execute(db._q("DELETE FROM jobs WHERE id = ?"), (job_id,))

    job_dir = Path(config.UPLOAD_DIR) / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)

    return {"status": "deleted"}


@app.get("/jobs")
def list_jobs():
    with db.get_db() as conn:
        jobs = conn.execute(
            db._q(
                "SELECT id, filename, status, total_pages, processed_pages, created_at "
                "FROM jobs ORDER BY created_at DESC"
            )
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


_MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/bmp": ".bmp",
    "image/x-ms-bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}


def run_job():
    result_fd = os.dup(1)
    os.dup2(2, 1)

    db.init_db()
    Path(config.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

    def _handle_stop(signum, frame):
        worker.stop()

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    content = sys.stdin.buffer.read()
    if not content:
        print("job: no input received on stdin", flush=True)
        sys.exit(2)

    mime = magic.from_buffer(content, mime=True)
    suffix = _MIME_TO_SUFFIX.get(mime)
    if suffix is None:
        print(f"job: unsupported input type ({mime})", flush=True)
        sys.exit(2)

    job_id = uuid.uuid4().hex
    job_dir = Path(config.UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True)
    (job_dir / f"input{suffix}").write_bytes(content)

    now = time.time()
    with db.get_db() as conn:
        conn.execute(
            db._q("INSERT INTO jobs (id, filename, status, created_at, updated_at) VALUES (?, ?, 'queued', ?, ?)"),
            (job_id, f"stdin{suffix}", now, now),
        )

    print(f"job: processing {job_id}", flush=True)
    ok = worker.process_file_job(job_id)
    if ok:
        with db.get_db() as conn:
            pages = conn.execute(
                db._q("SELECT markdown FROM pages WHERE job_id = ? ORDER BY page_num"),
                (job_id,),
            ).fetchall()
        markdown = "\n\n".join(p["markdown"] for p in pages)
        with os.fdopen(result_fd, "wb") as out:
            out.write(markdown.encode("utf-8"))
    print(f"job: {job_id} {'completed' if ok else 'failed'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if config.RUN_MODE == "job":
        run_job()
    else:
        uvicorn.run(app, host="0.0.0.0", port=8000)
