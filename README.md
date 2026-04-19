# Self-Hosted PDF OCR API for Large Documents

A self-hosted PDF OCR API powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and the PaddleOCR-VL model. Runs on GPU via Docker, processes PDFs page-by-page, and returns markdown content in JSON responses. Good support (not perfect) for Latvian and Lithuanian languages.

## Model

| | |
|---|---|
| **Model** | PaddleOCR-VL-1.5 |
| **Parameters** | 0.9B |
| **Layout detection** | PP-DocLayoutV3 |
| **GPU VRAM** | ~8.5GB |
| **Input formats** | PDF, PNG, JPG, JPEG, BMP, TIFF, WEBP |

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- NVIDIA GPU with ~8.5GB VRAM

## Quick start

**Using Docker Hub image:**

```yaml
services:
  paddleocr:
    image: edgaras0x4e/paddleocr-pdf-api:latest
    ports:
      - "8099:8000"
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
| `OCR_DPI` | `200` | DPI for PDF page rendering |
| `DB_PATH` | `/data/ocr.db` | SQLite database path |
| `UPLOAD_DIR` | `/data/uploads` | Upload storage path |

### Image descriptions (optional)

When enabled, cropped image regions (photos, charts, seals, logos) detected by the layout model are sent to an OpenAI-compatible vision model, and the returned description is inlined in the page markdown as a `> **[Label]** ...` blockquote. Disabled by default - the original behavior (stripping image tags) is preserved.

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_DESCRIPTION_ENABLED` | `false` | Master switch. When `false`, images are stripped as before. |
| `IMAGE_DESCRIPTION_PROVIDER` | `openai` | `openai` (any `OpenAI(base_url=…)`-compatible endpoint) or `azure` (uses `AzureOpenAI`). |
| `IMAGE_DESCRIPTION_API_URL` | `https://api.openai.com/v1` | Base URL. For `azure`, the resource endpoint, e.g. `https://<name>.cognitiveservices.azure.com`. |
| `IMAGE_DESCRIPTION_API_KEY` | _(empty)_ | Bearer / API key. Local backends accept any placeholder. |
| `IMAGE_DESCRIPTION_API_VERSION` | _(empty)_ | Azure-only, e.g. `2025-01-01-preview` (chat) or `2025-04-01-preview` (responses). |
| `IMAGE_DESCRIPTION_API_MODE` | `chat_completions` | `chat_completions` (universal) or `responses` (OpenAI-native / Azure). |
| `IMAGE_DESCRIPTION_MODEL` | `gpt-5.4` | Model name (or Azure deployment name). |
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
  - IMAGE_DESCRIPTION_MODEL="gpt-5.4"
  - IMAGE_DESCRIPTION_API_KEY="<your-key>"
  - IMAGE_DESCRIPTION_PROMPT_CHART="Extract all data points from this chart as a markdown table."
```

### Enabling API key authentication

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

## docker-compose.yml

```yaml
services:
  paddleocr:
    build: .
    ports:
      - "8099:8000"
    # environment:
    #   - API_KEY=your-secret-key
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

## How it works

1. A PDF is uploaded and saved to disk
2. A background worker picks up queued jobs in order
3. Each page is rendered to an image using pypdfium2
4. PaddleOCR-VL extracts text and converts it to markdown
5. HTML tags and image placeholders are stripped from the output
6. Results are stored in SQLite and available per-page as they complete
7. Jobs interrupted by a restart are automatically re-queued

## Data persistence

The `/data` volume stores the SQLite database and uploaded PDFs. This is a named Docker volume (`ocr-data`) that persists across container restarts and rebuilds.

## License

MIT

## Changelog

### v0.2.0

- Accept single image uploads (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`) as 1-page jobs.
- Optional image descriptions via OpenAI / Azure OpenAI models.
- Fixed tables to markdown tables.
