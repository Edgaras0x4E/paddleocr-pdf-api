# Self-Hosted PDF OCR API for Large Documents

A self-hosted PDF OCR API powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models (PaddleOCR-VL by default). Runs via Docker, processes PDFs page-by-page, and returns markdown content in JSON responses. Good support (not perfect) for Latvian and Lithuanian languages.

## Contents

- [OCR engines](#ocr-engines)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Usage](#usage)
- [API reference](#api-reference)
- [Configuration](#configuration)
  - [Database backend](#database-backend)
  - [Job mode](#job-mode)
  - [Image descriptions](#image-descriptions)
  - [Bounding boxes](#bounding-boxes)
  - [API key authentication](#api-key-authentication)
- [Data persistence](#data-persistence)
- [Changelog](#changelog)

## OCR engines

The API runs one of three OCR engines, selected with the `OCR_ENGINE` variable. All engines share the same API and job flow:

| `OCR_ENGINE` | Models | Output | Hardware |
|---|---|---|---|
| `vl` (default) | PaddleOCR-VL-1.6 (0.9B) + PP-DocLayoutV3 | Markdown, full document understanding | GPU, ~8.5GB VRAM |
| `text` | PP-OCRv5 server detection + recognition | Plain text lines, no layout markup | GPU or CPU |
| `structure` | PP-StructureV3 layout parsing (tables, formulas, reading order), no VLM | Markdown | GPU, ~10.5GB VRAM |

All engines accept the same input formats: PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP.

Each engine has its own baked image (models included, no first-run download):

| Image tag | Engine | Runs on |
|---|---|---|
| `latest-vl-baked` | `vl` | GPU |
| `latest-structure-baked` | `structure` | GPU |
| `latest-text-baked` | `text` | CPU |

Baked images preselect their engine, so `OCR_ENGINE` does not need to be set. The non-baked `latest` image is GPU-based and works with any engine; it downloads the selected engine's models at runtime.

## Requirements

Every image needs Docker. The GPU images also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and an NVIDIA GPU:

| Image | Needs |
|---|---|
| `latest` | Docker + NVIDIA GPU (for `vl` engine) |
| `latest-vl-baked` | Docker + NVIDIA GPU (~8.5GB VRAM) |
| `latest-structure-baked` | Docker + NVIDIA GPU (~10.5GB VRAM) |
| `latest-text-baked` | Docker |

## Quick start

**Using the [Docker Hub image](https://hub.docker.com/r/edgaras0x4e/paddleocr-pdf-api):**

```yaml
services:
  paddleocr:
    image: edgaras0x4e/paddleocr-pdf-api:latest
    ports:
      - "8099:8000"
    environment:
      - OCR_ENGINE=vl              # or text, or structure
    volumes:
      - ocr-data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

volumes:
  ocr-data:
```

```bash
docker compose up -d
```

**Or build from source:**

```bash
git clone https://github.com/Edgaras0x4E/paddleocr-pdf-api.git && cd paddleocr-pdf-api
docker compose up --build -d
```

The API will be available at `http://localhost:8099`. On first startup the model (~2GB) is downloaded and loaded into GPU memory. The API accepts requests immediately, but jobs will start processing once the model is ready.

**Baked images (no first-run download):** for scale-to-zero or cold-start-sensitive deployments, use your engine's baked tag instead (see [OCR engines](#ocr-engines)). The models are pre-baked into the image, so the container starts without downloading anything; cold-start warmup is only the model load into memory.

## Usage

### Submit a PDF

Also accepts single image files as a bonus: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp` (processed as a 1-page job).

```bash
curl -X POST http://localhost:8099/ocr -F "file=@document.pdf"
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "filename": "document.pdf",
  "status": "queued"
}
```

### Check progress

```bash
curl http://localhost:8099/ocr/{job_id}
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "filename": "document.pdf",
  "status": "processing",
  "total_pages": 185,
  "processed_pages": 42,
  "error": null
}
```

### Get a single page

```bash
curl http://localhost:8099/ocr/{job_id}/pages/1
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "page_num": 1,
  "markdown": "## Chapter 1\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit..."
}
```

### Get all pages

```bash
curl http://localhost:8099/ocr/{job_id}/result
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "filename": "document.pdf",
  "status": "completed",
  "total_pages": 185,
  "processed_pages": 185,
  "pages": [
    {"page_num": 1, "markdown": "## Chapter 1\n\nLorem ipsum dolor sit amet..."},
    {"page_num": 2, "markdown": "..."}
  ]
}
```

### List all jobs

```bash
curl http://localhost:8099/jobs
```

```json
{
  "jobs": [
    {
      "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
      "filename": "document.pdf",
      "status": "completed",
      "total_pages": 185,
      "processed_pages": 185
    }
  ]
}
```

### Cancel a job

```bash
curl -X POST http://localhost:8099/ocr/{job_id}/cancel
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "status": "cancelling"
}
```

### Delete a job

```bash
curl -X DELETE http://localhost:8099/ocr/{job_id}
```

```json
{
  "status": "deleted"
}
```

## API reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ocr` | Upload a PDF for processing |
| `GET` | `/ocr/{job_id}` | Get job status and progress |
| `GET` | `/ocr/{job_id}/pages/{page_num}` | Get markdown for a specific page |
| `GET` | `/ocr/{job_id}/result` | Get all completed pages |
| `POST` | `/ocr/{job_id}/cancel` | Cancel a queued or running job |
| `DELETE` | `/ocr/{job_id}` | Delete a job and its data |
| `GET` | `/jobs` | List all jobs |

## Configuration

Environment variables set in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_KEY` | _(empty)_ | Optional API key. When set, all requests must include an `X-API-Key` header |
| `OCR_ENGINE` | `vl` | OCR engine: `vl`, `text`, or `structure` (see [OCR engines](#ocr-engines)) |
| `OCR_DPI` | `200` | DPI for PDF page rendering |
| `OCR_MARKDOWN_IGNORE_LABELS` | `number, footnote, header, header_image, footer, footer_image, aside_text` | Comma-separated layout labels dropped from the markdown (`vl` and `structure` engines). |
| `OCR_BOUNDING_BOXES` | `false` | Adds a `bounding_boxes` object to page responses: page size and every detected block's label, pixel box and text (see [Bounding boxes](#bounding-boxes)). |
| `DATABASE_URL` | _(empty)_ | Empty uses SQLite at `DB_PATH`. A `postgresql://...` URL uses PostgreSQL instead. |
| `RUN_MODE` | `server` | `server` is the always-on API. `job` reads one file from stdin, processes it, and exits. |
| `DB_PATH` | `/data/ocr.db` | SQLite database path (used when `DATABASE_URL` is empty) |
| `UPLOAD_DIR` | `/data/uploads` | Upload storage path |

### Database backend

By default the API stores jobs and results in SQLite at `/data/ocr.db`, persisted on the `/data` volume. No configuration is needed.

To use PostgreSQL instead, set `DATABASE_URL`:

```yaml
environment:
  - DATABASE_URL=postgresql://user:password@host:5432/ocr
```

The schema is created automatically on startup. With PostgreSQL the database is external, so the API is stateless apart from the uploaded files held on `/data` during processing. Multiple server containers can share one PostgreSQL database.

### Job mode

By default the container runs as an always-on server (`RUN_MODE=server`). Set `RUN_MODE=job` to instead process a single file and exit. The file is piped in on stdin; the result is written to the same database as server mode and read back through the API:

```bash
docker run --rm -i --gpus all \
  -e RUN_MODE=job \
  edgaras0x4e/paddleocr-pdf-api:latest < document.pdf
```

The container reads the file from stdin, runs the same OCR pipeline, saves the result to the database, and exits (`0` on success, non-zero on failure). It does not start the HTTP server. The baked image variant avoids the model download on each run, which suits repeated one-shot jobs.

To keep the result, point the run at a persistent database:

```bash
# PostgreSQL (external DB, nothing mounted)
docker run --rm -i --gpus all \
  -e RUN_MODE=job -e DATABASE_URL=postgresql://user:password@host:5432/ocr \
  edgaras0x4e/paddleocr-pdf-api:latest-vl-baked < document.pdf

# SQLite (mount the /data volume so the database persists)
docker run --rm -i --gpus all \
  -e RUN_MODE=job -v ocr-data:/data \
  edgaras0x4e/paddleocr-pdf-api:latest-vl-baked < document.pdf
```

### Image descriptions

When enabled, cropped image regions (photos, charts, seals, logos) detected by the layout model are sent to an OpenAI-compatible vision model, and the returned description is inlined in the page markdown as a `> **[Label]** ...` blockquote. Disabled by default - the original behavior (stripping image tags) is preserved.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DESCRIPTION_ENABLED` | `false` | Master switch. When `false`, images are stripped as before. |
| `IMAGE_DESCRIPTION_PROVIDER` | `openai` | `openai` (any `OpenAI(base_url=…)`-compatible endpoint), `azure` (uses `AzureOpenAI`) or `anthropic` (Claude, uses the `anthropic` SDK; `IMAGE_DESCRIPTION_API_URL` and `IMAGE_DESCRIPTION_API_MODE` are not used). |
| `IMAGE_DESCRIPTION_API_URL` | `https://api.openai.com/v1` | Base URL. For `azure`, the resource endpoint, e.g. `https://<name>.cognitiveservices.azure.com`. |
| `IMAGE_DESCRIPTION_API_KEY` | _(empty)_ | Bearer / API key. Local backends accept any placeholder. |
| `IMAGE_DESCRIPTION_API_VERSION` | _(empty)_ | Azure-only, e.g. `2025-01-01-preview` (chat) or `2025-04-01-preview` (responses). |
| `IMAGE_DESCRIPTION_API_MODE` | `chat_completions` | `chat_completions` (universal) or `responses` (OpenAI-native / Azure). |
| `IMAGE_DESCRIPTION_MODEL` | `gpt-5.5` | Model name (or Azure deployment name). For `anthropic`, for example `claude-opus-5-5`. |
| `IMAGE_DESCRIPTION_PROMPT` | _built-in neutral prompt_ | Default prompt used when no per-label override is set. |
| `IMAGE_DESCRIPTION_PROMPT_<LABEL>` | _(empty)_ | Per-label override, e.g. `IMAGE_DESCRIPTION_PROMPT_CHART="Extract numeric data as a markdown table."`. Label is the uppercase `block_label` (`IMAGE`, `CHART`, `SEAL`, `HEADER_IMAGE`, `FOOTER_IMAGE`). |
| `IMAGE_DESCRIPTION_LABELS` | `image,chart,seal,header_image,footer_image` | Comma-separated labels to describe. `table` and `formula` are always skipped (PaddleOCR renders them natively). |
| `IMAGE_DESCRIPTION_MIN_PIXELS` | `10000` | Skip crops smaller than this area (w × h). Filters out bullet icons. |
| `IMAGE_DESCRIPTION_MAX_EDGE_PX` | `1568` | Downscale longest edge before sending. `0` disables. |
| `IMAGE_DESCRIPTION_MAX_PER_PAGE` | `10` | Cap of described images per page. |
| `IMAGE_DESCRIPTION_TIMEOUT` | `60` | Seconds per request. |
| `IMAGE_DESCRIPTION_MAX_RETRIES` | `2` | Retries on transient errors. |
| `IMAGE_DESCRIPTION_ON_ERROR` | `skip` | `skip`, `placeholder` (inserts `[image description unavailable]`), or `fail`. |

Example `docker-compose.yml` override:

```yaml
environment:
  - IMAGE_DESCRIPTION_ENABLED=true
  - IMAGE_DESCRIPTION_PROVIDER=azure
  - IMAGE_DESCRIPTION_API_URL="https://<your-resource>.cognitiveservices.azure.com"
  - IMAGE_DESCRIPTION_API_VERSION="2025-04-01-preview"
  - IMAGE_DESCRIPTION_API_MODE=responses
  - IMAGE_DESCRIPTION_MODEL="gpt-5.5"
  - IMAGE_DESCRIPTION_API_KEY="<your-key>"
  - IMAGE_DESCRIPTION_PROMPT_CHART="Extract all data points from this chart as a markdown table."
```

### Bounding boxes

Set `OCR_BOUNDING_BOXES=true` to get the position of every layout block. Page responses gain a `bounding_boxes` field:

```bash
curl http://localhost:8099/ocr/{job_id}/pages/1
```

```json
{
  "job_id": "994e7b398bb44d8ab5eade4d2ef57a15",
  "page_num": 1,
  "markdown": "## Chapter 1\n\nLorem ipsum dolor sit amet, consectetur adipiscing elit...",
  "bounding_boxes": {
    "width": 1654,
    "height": 2339,
    "dpi": 200,
    "blocks": [
      {"label": "paragraph_title", "bbox": [160, 210, 620, 262], "content": "Chapter 1"},
      {"label": "text", "bbox": [160, 300, 1490, 980], "content": "Lorem ipsum dolor sit amet, consectetur adipiscing elit..."},
      {"label": "table", "bbox": [160, 1040, 1490, 1620], "content": "<table><tr><td>...</td></tr></table>"},
      {"label": "image", "bbox": [160, 1700, 900, 2200], "content": ""}
    ]
  }
}
```

`bbox` is the position of the block: `[x1, y1, x2, y2]`. The unit is pixels of the rendered page. `width` and `height` are the size of that page. `dpi` is the resolution of that page. For an image upload, `dpi` is `null`. `content` is the text of the block. For a table, it is HTML. For an image, it is empty. The list includes the blocks that `OCR_MARKDOWN_IGNORE_LABELS` removes from the markdown. For a page processed before you set `OCR_BOUNDING_BOXES`, `bounding_boxes` is `null`.

The `text` engine returns one block for each text line:

```json
{"label": "text_line", "bbox": [x1, y1, x2, y2], "content": "the text of the line", "score": 0.99}
```

### API key authentication

Uncomment the environment section in `docker-compose.yml`:

```yaml
environment:
  - API_KEY=your-secret-key
```

Then restart:

```bash
docker compose down && docker compose up -d
```

All requests must then include the header:

```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8099/jobs
```

## Data persistence

The `/data` volume stores the SQLite database and uploaded PDFs. This is a named Docker volume (`ocr-data`) that persists across container restarts and rebuilds. When `DATABASE_URL` points at PostgreSQL, the database lives in PostgreSQL and `/data` only holds uploaded files during processing.

## Changelog

### v0.5.0

- Added `OCR_BOUNDING_BOXES` to return the position of each block in the page responses (see [Bounding boxes](#bounding-boxes)). Works with the `vl`, `text` and `structure` engines. Disabled by default.
- Added `anthropic` as an `IMAGE_DESCRIPTION_PROVIDER`, for image descriptions with Claude models.

### v0.4.1

- Added `OCR_MARKDOWN_IGNORE_LABELS` to control which layout labels are dropped from the markdown output. ([#1](https://github.com/Edgaras0x4E/paddleocr-pdf-api/issues/1))
- Added `OCR_ENGINE` to select the OCR engine: `vl` (default), `text`, or `structure`, each with its own baked image. ([#1](https://github.com/Edgaras0x4E/paddleocr-pdf-api/issues/1))
- Renamed the baked image to `:latest-vl-baked` / `:v0.4.1-vl-baked` (was `:latest-baked` / `:v0.4.0-baked`); `latest-baked` remains an alias.

### v0.4.0

- Added an optional PostgreSQL backend, selected with `DATABASE_URL`. SQLite remains default.
- Added an optional `job` run mode (`RUN_MODE=job`): the container reads a file, processes it, saves the result to the database, and exits.

### v0.3.0

- Updated the OCR model to PaddleOCR-VL-1.6 (requires `paddleocr` 3.6.0).
- Added a baked image variant (`:latest-baked` / `:v0.3.0-baked`) with the model pre-baked into the image - no first-run download, for fast cold starts and scale-to-zero deployments.

### v0.2.0

- Accept single image uploads (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`) as 1-page jobs.
- Optional image descriptions via OpenAI / Azure OpenAI models.
- Fixed tables to markdown tables.

## License

MIT
