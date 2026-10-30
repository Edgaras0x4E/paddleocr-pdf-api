import json
import os
import tempfile
import threading
import time
import traceback
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from paddleocr import PaddleOCR, PaddleOCRVL, PPStructureV3

from app import config
from app import db
from app.image_describe import describe_images
from app.markdown import convert_html_tables, strip_html, strip_image_tags

if config.BACKEND == "postgres":
    import psycopg


RECONNECT_BACKOFF_START = 2.0
RECONNECT_BACKOFF_MAX = 30.0


def _pixel_box(coords):
    return [int(round(float(v))) for v in coords]


class OCRWorker:
    def __init__(self):
        self._thread = None
        self._stop = threading.Event()
        self._model = None
        self._pg = None
        self._saw_failed_job = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def join(self):
        if self._thread is not None:
            self._thread.join()

    def close(self):
        self._reset_pg_conn()

    def _load_model(self):
        if self._model is None:
            engine = config.OCR_ENGINE
            print(f"Loading OCR engine '{engine}'...")
            if engine == "vl":
                self._model = PaddleOCRVL(
                    pipeline_version=config.PIPELINE_VERSION,
                    markdown_ignore_labels=config.MARKDOWN_IGNORE_LABELS,
                )
            elif engine == "structure":
                self._model = PPStructureV3(
                    markdown_ignore_labels=config.MARKDOWN_IGNORE_LABELS,
                )
            else:
                self._model = PaddleOCR()
            print("Model loaded.")
        return self._model

    def _interruptible_sleep(self, seconds):
        self._stop.wait(seconds)

    def _run(self):
        ocr = self._load_model()
        if config.BACKEND == "postgres":
            self._loop_postgres(ocr)
        else:
            self._loop_sqlite(ocr)

    def process_file_job(self, job_id):
        ocr = self._load_model()
        now = time.time()
        if config.BACKEND == "postgres":
            if not self._ensure_pg_conn():
                return False
            db.try_advisory_lock(self._pg, job_id)
            try:
                self._pg.execute(
                    db._q("UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ? AND status = 'queued'"),
                    (now, job_id),
                )
            except Exception:
                traceback.print_exc()
                self._reset_pg_conn()
                return False
        else:
            with db.get_db() as conn:
                conn.execute(
                    db._q("UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ? AND status = 'queued'"),
                    (now, job_id),
                )
        self._process_job(ocr, {"id": job_id})
        self._reset_pg_conn()
        return self._final_status(job_id) == "completed"

    def _final_status(self, job_id):
        with db.get_db() as conn:
            row = conn.execute(db._q("SELECT status FROM jobs WHERE id = ?"), (job_id,)).fetchone()
        return row["status"] if row else None

    def _loop_sqlite(self, ocr):
        backoff = RECONNECT_BACKOFF_START
        while not self._stop.is_set():
            try:
                job = db.claim_next_job_sqlite()
            except Exception:
                traceback.print_exc()
                self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
                continue
            backoff = RECONNECT_BACKOFF_START
            if job is None:
                self._interruptible_sleep(1)
                continue
            self._process_job(ocr, job)

    def _loop_postgres(self, ocr):
        backoff = RECONNECT_BACKOFF_START
        while not self._stop.is_set():
            if not self._ensure_pg_conn():
                self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
                continue
            try:
                job = db.claim_or_adopt_job_postgres(self._pg)
            except Exception:
                traceback.print_exc()
                self._reset_pg_conn()
                self._interruptible_sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)
                continue
            backoff = RECONNECT_BACKOFF_START
            if job is None:
                self._interruptible_sleep(1)
                continue
            self._process_job(ocr, job)

    def _ensure_pg_conn(self):
        if self._pg is not None and not self._pg.closed:
            return True
        self._reset_pg_conn()
        try:
            self._pg = db.connect_worker_postgres()
            return True
        except Exception:
            traceback.print_exc()
            return False

    def _reset_pg_conn(self):
        if self._pg is not None:
            try:
                self._pg.close()
            except Exception:
                pass
            self._pg = None

    def _is_conn_error(self, exc):
        if self._pg is None or self._pg.closed:
            return True
        return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError))

    def _with_conn(self, fn):
        if config.BACKEND == "postgres":
            return fn(self._pg)
        with db.get_db() as conn:
            return fn(conn)

    def _read_total_pages(self, job_id):
        def op(conn):
            row = conn.execute(db._q("SELECT total_pages FROM jobs WHERE id = ?"), (job_id,)).fetchone()
            return row["total_pages"] if row else 0
        return self._with_conn(op)

    def _read_done_pages(self, job_id):
        def op(conn):
            rows = conn.execute(db._q("SELECT page_num FROM pages WHERE job_id = ?"), (job_id,)).fetchall()
            return {r["page_num"] for r in rows}
        return self._with_conn(op)

    def _read_cancel_requested(self, job_id):
        def op(conn):
            row = conn.execute(db._q("SELECT cancel_requested FROM jobs WHERE id = ?"), (job_id,)).fetchone()
            return row["cancel_requested"] if row else 0
        return self._with_conn(op)

    def _write_total_pages(self, job_id, total_pages):
        now = time.time()

        def op(conn):
            conn.execute(
                db._q("UPDATE jobs SET total_pages = ?, updated_at = ? WHERE id = ? AND total_pages = 0"),
                (total_pages, now, job_id),
            )
        self._with_conn(op)

    def _write_page(self, job_id, page_num, markdown, bounding_boxes=None):
        now = time.time()

        def op(conn):
            conn.execute(
                db._q(
                    "INSERT INTO pages (job_id, page_num, markdown, bounding_boxes, created_at) "
                    "VALUES (?, ?, ?, ?, ?) ON CONFLICT (job_id, page_num) DO NOTHING"
                ),
                (job_id, page_num, markdown, bounding_boxes, now),
            )
            row = conn.execute(
                db._q(
                    "UPDATE jobs SET processed_pages = "
                    "(SELECT COUNT(DISTINCT page_num) FROM pages WHERE job_id = ?), updated_at = ? "
                    "WHERE id = ? RETURNING cancel_requested"
                ),
                (job_id, now, job_id),
            ).fetchone()
            return row["cancel_requested"] if row else 0
        return self._with_conn(op)

    def _write_completed(self, job_id):
        now = time.time()

        def op(conn):
            cur = conn.execute(
                db._q(
                    "UPDATE jobs SET status = 'completed', updated_at = ? "
                    "WHERE id = ? AND status = 'processing' AND cancel_requested = 0 "
                    "AND (SELECT COUNT(DISTINCT page_num) FROM pages WHERE job_id = ?) = total_pages"
                ),
                (now, job_id, job_id),
            )
            return cur.rowcount
        return self._with_conn(op)

    def _write_cancelled(self, job_id):
        now = time.time()

        def op(conn):
            conn.execute(
                db._q("UPDATE jobs SET status = 'cancelled', updated_at = ? WHERE id = ? AND status = 'processing'"),
                (now, job_id),
            )
        self._with_conn(op)

    def _write_failed(self, job_id, error):
        now = time.time()

        def op(conn):
            conn.execute(
                db._q(
                    "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? "
                    "WHERE id = ? AND status = 'processing'"
                ),
                (error, now, job_id),
            )
        self._with_conn(op)

    def _process_job(self, ocr, job):
        job_id = job["id"]
        try:
            self._run_job_pages(ocr, job_id)
        except Exception as e:
            if config.BACKEND == "postgres" and self._is_conn_error(e):
                traceback.print_exc()
                self._reset_pg_conn()
                return
            traceback.print_exc()
            if self._record_failure(job_id, str(e)):
                self._saw_failed_job = True
        finally:
            if config.BACKEND == "postgres" and self._pg is not None:
                db.advisory_unlock(self._pg, job_id)

    def _record_failure(self, job_id, error):
        try:
            self._write_failed(job_id, error)
            return True
        except Exception:
            traceback.print_exc()
            if config.BACKEND == "postgres":
                self._reset_pg_conn()
            return False

    def _open_input(self, input_path, is_pdf):
        if is_pdf:
            pdf = pdfium.PdfDocument(str(input_path))
            return pdf, len(pdf)
        return None, 1

    def _run_job_pages(self, ocr, job_id):
        job_dir = Path(config.UPLOAD_DIR) / job_id
        input_candidates = list(job_dir.glob("input.*"))
        if not input_candidates:
            raise FileNotFoundError(f"no input file in {job_dir}")
        input_path = input_candidates[0]
        is_pdf = input_path.suffix.lower() == ".pdf"

        total_pages = self._read_total_pages(job_id)
        if total_pages == 0:
            pdf, total_pages = self._open_input(input_path, is_pdf)
            if total_pages == 0:
                self._write_completed(job_id)
                print(f"[{job_id[:8]}] Job completed (0 pages)")
                return
            self._write_total_pages(job_id, total_pages)
        else:
            pdf, _ = self._open_input(input_path, is_pdf)

        done = self._read_done_pages(job_id)
        scale = config.DPI / 72

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            if self._stop.is_set():
                return
            if page_num in done:
                continue

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
                result = list(ocr.predict(input=tmp_path))
                if config.OCR_ENGINE == "text":
                    page_markdown = self._text_lines(result)
                else:
                    page_markdown = self._markdown_text(result, page_num, job_id)
                bounding_boxes = None
                if config.BOUNDING_BOXES:
                    bounding_boxes = json.dumps(
                        self._bounding_boxes(result, pil_image.size, config.DPI if is_pdf else None),
                        ensure_ascii=False,
                    )

                cancel_requested = self._write_page(job_id, page_num, page_markdown, bounding_boxes)
                print(f"[{job_id[:8]}] Page {page_num}/{total_pages} done")

                if cancel_requested:
                    self._write_cancelled(job_id)
                    print(f"[{job_id[:8]}] Job cancelled at page {page_num}/{total_pages}")
                    return

            finally:
                os.unlink(tmp_path)

        rc = self._write_completed(job_id)
        if rc == 0 and self._read_cancel_requested(job_id):
            self._write_cancelled(job_id)
        print(f"[{job_id[:8]}] Job completed ({total_pages} pages)")

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

    def _markdown_text(self, result, page_num, job_id):
        markdown_parts = []
        for res in result:
            md_data = res._to_markdown(pretty=False)
            if isinstance(md_data, dict):
                text = md_data.get("markdown_texts") or md_data.get("markdown") or ""
                images = md_data.get("markdown_images") or {}
                if text:
                    if config.IMAGE_DESCRIPTION_ENABLED and images:
                        text = describe_images(
                            text, images,
                            page_num=page_num, job_id=job_id,
                        )
                    else:
                        text = strip_image_tags(text)
                    markdown_parts.append(text)

        if not markdown_parts:
            markdown_parts = [self._extract_text(result)]

        page_markdown = "\n\n".join(markdown_parts)
        page_markdown = convert_html_tables(page_markdown)
        page_markdown = strip_html(page_markdown)
        return page_markdown

    def _text_lines(self, result):
        parts = []
        for res in result:
            lines = res.get("rec_texts") or []
            if lines:
                parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def _bounding_boxes(self, result, size, dpi):
        width, height = size
        blocks = []
        for res in result:
            if config.OCR_ENGINE == "text":
                texts = res.get("rec_texts")
                boxes = res.get("rec_boxes")
                scores = res.get("rec_scores")
                if texts is None or boxes is None:
                    continue
                for i, text in enumerate(texts):
                    block = {"label": "text_line", "bbox": _pixel_box(boxes[i]), "content": text}
                    if scores is not None:
                        block["score"] = round(float(scores[i]), 4)
                    blocks.append(block)
            else:
                for block in res.get("parsing_res_list") or []:
                    blocks.append({
                        "label": block.label,
                        "bbox": _pixel_box(block.bbox),
                        "content": block.content,
                    })
        return {"width": width, "height": height, "dpi": dpi, "blocks": blocks}
