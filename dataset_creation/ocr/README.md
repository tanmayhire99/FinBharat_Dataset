# OCR Pipeline

This directory contains the highly-optimized batch OCR and rendering pipeline used to convert Indian Annual Reports (PDFs) into structured Markdown format. 

## Overview
The pipeline is designed to process thousands of complex financial PDFs efficiently, specifically optimized for high-end hardware (e.g., RTX3090, A6000, A100, H100 GPUs)

## Key Features
- **docext Image Pipeline:** Fast, reliable PDF-to-image extraction and resizing.
- **Tesseract OSD Rotation:** Automatically detects and corrects page orientation (90, 180, 270 degrees).
- **vLLM Streaming API:** Uses `nanonets/Nanonets-OCR2-3B` and `nanonets/Nanonets-OCR-s` via vLLM for ultra-fast, timeout-resistant inference.
- **Strict CPU Throttling:** Enforces a physical OS-level CPU affinity mask and OpenMP thread limits to guarantee the script and all its sub-processes never exceed 8 CPU cores.
- **Inline Retry & Fallbacks:** Automatically retries failed pages and inserts placeholders for manual review if permanent failure occurs.

## Pipeline Architecture & Workflow
The OCR extraction is broken down into a multi-stage, highly parallelized workflow designed for fault tolerance and maximum throughput.

### 1. Pre-Processing & Queueing
- **Corrupt File Filtering:** Automatically scans the `data/` directory and drops files under 10KB to prevent ghost-processing errors.
- **State Management:** Checks `markdown/` output and `failed_conversions.csv` to resume seamlessly without re-processing completed documents.

### 2. CPU-Bound Image Extraction (Strictly 8-Core)
- **Docext Conversion:** Converts PDF pages to high-resolution JPEGs.
- **Tesseract OSD:** Uses Optical Character Recognition to detect text orientation.
- **Auto-Rotation & Resizing:** Uses OpenCV to physically rotate upside-down pages and resize them to a standardized 2048px. 
*Note: This entire phase is fenced into a 7-permit Python Semaphore and hard OS-level CPU affinity mask. This guarantees these heavy C++ libraries only touch exactly 8 CPU logical cores, preventing the system from freezing.*

### 3. GPU-Bound OCR Inference
- **Concurrent Dispatch:** The rotated images are base64-encoded and dispatched via concurrent HTTP threads (`PARALLEL_WORKERS = 180`) to the background vLLM server.
- **Nanonets Processing:** The vision-language model reads the image "naturally", formatting tables as raw HTML, equations as LaTeX, and wrapping watermarks/page numbers in specific XML tags.
- **Streaming Iteration:** Uses HTTP chunked streaming (receiving tokens one by one) to prevent network connection timeouts on extremely dense, difficult pages.

### 4. Post-Processing & Stitching
- **Asynchronous Assembly:** As the asynchronous page workers return markdown strings, they are stored in memory. Once an entire PDF is completed, the pages are stitched together in numerical order separated by `\n\n---\n\n`.
- **Multi-Pass Retry System:** If a page fails (network timeout, vLLM CUDA OOM error), it is marked with an `[EXTRACTION FAILED]` placeholder. After finishing the full batch of PDFs, the script executes a "Retry Pass" specifically targeting and re-processing only the failed placeholder pages.

## Prerequisites
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
# Requires: vllm, docext, opencv-python, pytesseract, requests, tqdm, psutil
```

## Configuration
The main configuration is hardcoded in `batch_pdf2md.py`. Key parameters include:
- **Model**: `nanonets/Nanonets-OCR2-3B` and `nanonets/Nanonets-OCR-s` (vLLM)
- **Parallelism**: ** total workers, ** PDFs in parallel, ** vLLM sequences
- **CPU Affinity**: Hardcoded to 8 logical cores (0-7)

## Usage
Run the script from the `ocr` directory:

```bash
nohup python batch_pdf2md.py > nohup.out 2>&1 &
tail -f nohup.out
```

## Output
- Markdown files are saved to `../markdown/`
- Logs are written to `../conversion.log`
- Progress is tracked in `../progress.txt`
- Failed conversions are listed in `../failed_conversions.csv`

## Important Notes
1.  **vLLM Server**: Ensure a vLLM server running the specified model is accessible on port 8000 before starting.
2.  **Temporary Files**: The script uses a cache directory `.page_cache/` for intermediate image files.
