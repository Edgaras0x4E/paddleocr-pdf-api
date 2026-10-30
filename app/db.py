import sqlite3
import time
import zlib
from contextlib import contextmanager
from pathlib import Path

from app import config

if config.BACKEND == "postgres":
    import psycopg
    from psycopg.rows import dict_row


ADVISORY_LOCK_CLASS_ID = 0x4F435231


def _q(sql: str) -> str:
    if config.BACKEND == "postgres":
        return sql.replace("?", "%s")
    return sql


@contextmanager
def get_db():
    if config.BACKEND == "postgres":
        conn = psycopg.connect(config.DATABASE_URL, row_factory=dict_row)
    else:
        conn = sqlite3.connect(config.DB_PATH, timeout=30)
        conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def connect_worker_postgres():
    return psycopg.connect(config.DATABASE_URL, autocommit=True, row_factory=dict_row)


def _advisory_key(job_id: str):
    h = zlib.crc32(job_id.encode("utf-8")) & 0xFFFFFFFF
    if h >= 0x80000000:
        h -= 0x100000000
    return ADVISORY_LOCK_CLASS_ID, h


def try_advisory_lock(conn, job_id: str) -> bool:
    cls, key = _advisory_key(job_id)
    row = conn.execute("SELECT pg_try_advisory_lock(%s, %s) AS locked", (cls, key)).fetchone()
    return bool(row["locked"])


def advisory_unlock(conn, job_id: str):
    if conn is None:
        return
    cls, key = _advisory_key(job_id)
    try:
        conn.execute("SELECT pg_advisory_unlock(%s, %s)", (cls, key))
    except Exception:
        pass


def init_db():
    if config.BACKEND == "postgres":
        _init_db_postgres()
    else:
        _init_db_sqlite()


def _init_db_sqlite():
    Path(config.DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    with get_db() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_pages INTEGER DEFAULT 0,
                processed_pages INTEGER DEFAULT 0,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(id),
                page_num INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(job_id, page_num)
            )
        """)
        cols = [r["name"] for r in db.execute("PRAGMA table_info(jobs)").fetchall()]
        if "cancel_requested" not in cols:
            db.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        page_cols = [r["name"] for r in db.execute("PRAGMA table_info(pages)").fetchall()]
        if "bounding_boxes" not in page_cols:
            db.execute("ALTER TABLE pages ADD COLUMN bounding_boxes TEXT")


def _init_db_postgres():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_pages INTEGER DEFAULT 0,
                processed_pages INTEGER DEFAULT 0,
                error TEXT,
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS pages (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                page_num INTEGER NOT NULL,
                markdown TEXT NOT NULL,
                created_at DOUBLE PRECISION NOT NULL,
                UNIQUE(job_id, page_num)
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_jobs_active
            ON jobs (created_at) WHERE status IN ('queued', 'processing')
        """)
        exists = db.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'jobs' "
            "AND column_name = 'cancel_requested'"
        ).fetchone()
        if not exists:
            db.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")
        db.execute("ALTER TABLE pages ADD COLUMN IF NOT EXISTS bounding_boxes TEXT")


def recover_sqlite_jobs():
    now = time.time()
    with get_db() as db:
        rows = db.execute("SELECT id FROM jobs WHERE status = 'processing'").fetchall()
        for row in rows:
            db.execute(
                "UPDATE jobs SET processed_pages = "
                "(SELECT COUNT(DISTINCT page_num) FROM pages WHERE job_id = ?) WHERE id = ?",
                (row["id"], row["id"]),
            )
        db.execute(
            "UPDATE jobs SET status = 'queued', updated_at = ? WHERE status = 'processing'",
            (now,),
        )


def claim_next_job_sqlite():
    with get_db() as db:
        row = db.execute(
            "SELECT id, filename FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
        ).fetchone()
        if not row:
            return None
        cur = db.execute(
            "UPDATE jobs SET status = 'processing', updated_at = ? WHERE id = ? AND status = 'queued'",
            (time.time(), row["id"]),
        )
        if cur.rowcount == 1:
            return dict(row)
    return None


def claim_or_adopt_job_postgres(conn):
    adopted = _adopt_abandoned_job_postgres(conn)
    if adopted is not None:
        return adopted
    return _claim_queued_job_postgres(conn)


def _adopt_abandoned_job_postgres(conn):
    rows = conn.execute(
        "SELECT id FROM jobs WHERE status = 'processing' ORDER BY created_at"
    ).fetchall()
    for row in rows:
        job_id = row["id"]
        if not try_advisory_lock(conn, job_id):
            continue
        current = conn.execute(
            "SELECT id, filename, status FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        if current is not None and current["status"] == "processing":
            return {"id": current["id"], "filename": current["filename"]}
        advisory_unlock(conn, job_id)
    return None


def _claim_queued_job_postgres(conn):
    rows = conn.execute(
        "SELECT id, filename FROM jobs WHERE status = 'queued' ORDER BY created_at"
    ).fetchall()
    for row in rows:
        job_id = row["id"]
        if not try_advisory_lock(conn, job_id):
            continue
        cur = conn.execute(
            "UPDATE jobs SET status = 'processing', updated_at = %s WHERE id = %s AND status = 'queued'",
            (time.time(), job_id),
        )
        if cur.rowcount == 1:
            return {"id": job_id, "filename": row["filename"]}
        advisory_unlock(conn, job_id)
    return None
