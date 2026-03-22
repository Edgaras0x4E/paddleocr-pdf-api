# Self-Hosted PDF OCR API for Large Documents

A self-hosted PDF OCR API powered by [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and the PaddleOCR-VL model. Runs on GPU via Docker, processes PDFs page-by-page, and returns markdown content in JSON responses. Good support (not perfect) for Latvian and Lithuanian languages.

## Model

| | |
|---|---|
| **Model** | PaddleOCR-VL-1.5 |
| **Parameters** | 0.9B |
| **Layout detection** | PP-DocLayoutV3 |
| **GPU VRAM** | ~8.5GB |

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
