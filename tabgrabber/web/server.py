"""Local web UI for TabGrabber.

Serves a single page where you paste a YouTube or Spotify link (or upload a
file), pick options, and watch the pipeline run. When it finishes you get a
player with the chord sheet synced to the audio, alongside the generated tabs.

Start it with ``tabgrabber-web`` or ``python -m tabgrabber.web.server``.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import re
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from tabgrabber.cli import parse_tuning
from tabgrabber.pipeline import PipelineOptions, process
from tabgrabber.web.chordsheet import DEFAULT_LYRICS_MODEL, build_chordsheet

logger = logging.getLogger("tabgrabber")

PACKAGE_DIR = Path(__file__).resolve().parent

DEFAULT_PORT = 8420
DEFAULT_DATA_DIR = "tabgrabber-web-data"

# Uploads are capped so a stray file cannot fill the disk, and restricted to
# the container formats the pipeline can actually read.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}

# Finished jobs are kept only so the browser can re-read their results; the
# oldest are dropped rather than growing the process forever.
MAX_TRACKED_JOBS = 40

SPOTIFY_TRACK_RE = re.compile(
    r"open\.spotify\.com/(?:intl-\w+/)?track/([A-Za-z0-9]+)", re.I)


# --------------------------------------------------------------------- paths


@dataclass
class Workspace:
    """Directories the server reads and writes."""
    root: Path

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def uploads(self) -> Path:
        return self.root / "uploads"

    @property
    def output(self) -> Path:
        return self.root / "output"

    def create(self) -> "Workspace":
        for directory in (self.downloads, self.uploads, self.output):
            directory.mkdir(parents=True, exist_ok=True)
        return self


workspace = Workspace(Path.cwd() / DEFAULT_DATA_DIR)


# ----------------------------------------------------------------- job state


@dataclass
class Job:
    """One end-to-end run, tracked so the browser can follow along."""
    id: str
    status: str = "queued"       # queued|downloading|processing|done|error
    progress: float = 0.0        # 0-1, meaningful only while downloading
    title: str = ""
    error: str = ""
    output_dir: Path | None = None
    source_audio: Path | None = None   # served via /api/audio/{job_id}
    sheet_rel: str = ""                # chordsheet.json, relative to output/
    logs: list[str] = field(default_factory=list)
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)

    def emit(self, kind: str, **payload: Any) -> None:
        if kind == "log":
            self.logs.append(payload.get("line", ""))
            del self.logs[:-2000]
        self.events.put({"kind": kind, **payload})


jobs: "OrderedDict[str, Job]" = OrderedDict()
# The pipeline is GPU-bound, so only one job runs at a time.
pipeline_lock = threading.Lock()


def _register(job: Job) -> None:
    jobs[job.id] = job
    while len(jobs) > MAX_TRACKED_JOBS:
        jobs.popitem(last=False)


def _get_job(job_id: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job")
    return job


class JobLogHandler(logging.Handler):
    """Forward tabgrabber log records into one job's event stream."""

    def __init__(self, job: Job) -> None:
        super().__init__()
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.job.emit("log", line=self.format(record), level=record.levelname)
        except Exception:  # pragma: no cover - logging must never raise
            pass


# ------------------------------------------------------------- link handling


def spotify_search_query(url: str, job: Job) -> str:
    """Turn a Spotify track link into an "Artist - Title" search query.

    Spotify streams are DRM protected and cannot be downloaded, so the public
    track metadata is read and the same song is looked up on YouTube instead.
    """
    import urllib.request

    match = SPOTIFY_TRACK_RE.search(url)
    if not match:
        raise ValueError("Not a Spotify track link (expected /track/...)")

    track_url = f"https://open.spotify.com/track/{match.group(1)}"
    job.emit("log", line=f"Reading Spotify metadata: {track_url}", level="INFO")

    request = urllib.request.Request(
        track_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        html = response.read().decode("utf-8", errors="replace")

    def meta(prop: str) -> str:
        for pattern in (
            rf'<meta[^>]+property=["\']{re.escape(prop)}["\'][^>]+content=["\']([^"\']+)["\']',
            rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']{re.escape(prop)}["\']',
        ):
            found = re.search(pattern, html)
            if found:
                return found.group(1)
        return ""

    title = meta("og:title").strip()
    if not title:
        raise ValueError("Could not read the track title from Spotify")

    # og:description is usually "Artist - Album - Song - Year", dot separated.
    artist = ""
    for part in (p.strip() for p in meta("og:description").split("·")):
        if part and part.lower() != title.lower() and not part.isdigit():
            artist = part
            break

    query = f"{artist} - {title}".strip(" -")
    job.emit("log", line=f"Spotify track: {query} - searching YouTube",
             level="INFO")
    return query


def download_audio(source: str, job: Job) -> Path:
    """Fetch audio for a URL or a search phrase with yt-dlp."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError(
            "Downloading links needs the 'web' extra: "
            "pip install 'tabgrabber[web]'"
        ) from exc

    if SPOTIFY_TRACK_RE.search(source):
        target = "ytsearch1:" + spotify_search_query(source, job)
    elif source.startswith(("http://", "https://")):
        target = source
    else:
        target = "ytsearch1:" + source

    job.status = "downloading"
    job.emit("status", status="downloading", stage="Downloading audio")

    reported = -1.0

    def on_progress(state: dict) -> None:
        nonlocal reported
        if state.get("status") == "downloading":
            total = state.get("total_bytes") or state.get("total_bytes_estimate")
            done = state.get("downloaded_bytes") or 0
            if total:
                fraction = done / total
                if fraction - reported >= 0.02:
                    reported = fraction
                    job.progress = fraction
                    job.emit("progress", progress=fraction,
                             stage=f"Downloading audio {fraction * 100:.0f}%")
        elif state.get("status") == "finished":
            job.progress = 1.0
            job.emit("progress", progress=1.0, stage="Converting to MP3")

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(workspace.downloads / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [on_progress],
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(target, download=True)
        if info.get("_type") == "playlist":
            info = info["entries"][0]
        job.title = info.get("title", "")
        job.emit("log", line=f"Downloaded: {job.title}", level="INFO")
        downloaded = Path(ydl.prepare_filename(info))

    converted = downloaded.with_suffix(".mp3")
    if converted.exists():
        return converted
    if downloaded.exists():
        return downloaded
    # The postprocessor may have renamed it; fall back to the newest MP3.
    candidates = sorted(workspace.downloads.glob("*.mp3"),
                        key=lambda p: p.stat().st_mtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("Could not locate the downloaded audio")


# ------------------------------------------------------------------ pipeline


def safe_name(name: str) -> str:
    """Filesystem-safe directory name for a song title."""
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return cleaned[:120] or "song"


def find_vocal_stem(output_dir: Path) -> Path | None:
    matches = sorted((output_dir / "stems").rglob("vocals.wav"))
    return matches[0] if matches else None


def write_chordsheet(job: Job, request: "JobRequest", result: Any,
                     output_dir: Path) -> str:
    """Write chordsheet.json; returns its path relative to output/."""
    if getattr(result, "analysis", None) is None:
        logger.warning("No song analysis available, skipping the chord sheet")
        return ""

    if request.lyrics:
        job.emit("status", status="processing", stage="Transcribing lyrics")

    try:
        sheet = build_chordsheet(
            analysis=result.analysis,
            vocals_path=find_vocal_stem(output_dir) if request.lyrics else None,
            model_size=request.lyrics_model,
            device=request.device,
            language=request.lyrics_language.strip() or None,
        )
    except Exception as exc:  # noqa: BLE001 - the sheet is a bonus, not the job
        logger.error(f"Could not build the chord sheet: {exc}")
        return ""

    path = output_dir / "chordsheet.json"
    path.write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Wrote {path.name}: {len(sheet['lines'])} lines, "
                f"{len(sheet['chords'])} chords")
    return path.relative_to(workspace.output).as_posix()


def run_job(job: Job, request: "JobRequest", input_path: Path | None) -> None:
    handler = JobLogHandler(job)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    level = logging.DEBUG if request.verbose else logging.INFO
    handler.setLevel(level)

    try:
        if input_path is None:
            input_path = download_audio(request.source, job)
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file is missing: {input_path}")
        job.source_audio = input_path

        with pipeline_lock:
            job.status = "processing"
            job.emit("status", status="processing",
                     stage="Separating stems and analysing")

            output_dir = workspace.output / safe_name(job.title or input_path.stem)
            if output_dir.exists():
                output_dir = output_dir.with_name(
                    f"{output_dir.name}_{int(time.time())}")
            output_dir.mkdir(parents=True, exist_ok=True)
            job.output_dir = output_dir

            options = PipelineOptions.from_preset(
                request.quality,
                device=request.device,
                instruments=request.instruments,
                formats=request.formats,
                tuning=parse_tuning(request.tuning) if request.tuning.strip() else None,
                onset_threshold=request.onset_threshold,
                frame_threshold=request.frame_threshold,
                invert_strings=request.invert_strings,
                # The web UI transcribes lyrics itself, via the chord sheet.
                lyrics_disabled=True,
            )

            logger.setLevel(level)
            logger.addHandler(handler)
            try:
                result = process(input_path, output_dir, options)
                job.sheet_rel = write_chordsheet(job, request, result, output_dir)
            finally:
                logger.removeHandler(handler)

        job.status = "done"
        job.emit("status", status="done", stage="Finished")
        job.emit("done", files=collect_files(job),
                 audio=f"/api/audio/{job.id}", chordsheet=job.sheet_rel)

    except Exception as exc:  # noqa: BLE001 - surfaced to the browser
        job.status = "error"
        job.error = str(exc)
        job.emit("log", line=traceback.format_exc(), level="ERROR")
        job.emit("status", status="error", stage="Failed", error=str(exc))
    finally:
        job.events.put({"kind": "eof"})


def collect_files(job: Job) -> list[dict]:
    """Everything the run produced, grouped for display."""
    if not job.output_dir or not job.output_dir.exists():
        return []

    files = []
    for path in sorted(job.output_dir.rglob("*")):
        if not path.is_file():
            continue
        parts = path.relative_to(job.output_dir).parts
        top = parts[0] if len(parts) > 1 else ""
        if top in ("tabs", "midi", "stems"):
            category = top
        elif path.suffix == ".wav":
            category = "stems"
        else:
            category = "other"
        files.append({
            "name": path.name,
            "path": path.relative_to(workspace.output).as_posix(),
            "category": category,
            "size": path.stat().st_size,
            "preview": path.suffix == ".txt",
        })
    return files


# ----------------------------------------------------------------------- API


class JobRequest(BaseModel):
    source: str = ""
    upload_path: str = ""
    device: str = "auto"
    quality: str = "fast"
    instruments: list[str] = ["guitar", "bass"]
    formats: list[str] = ["ascii"]
    tuning: str = ""
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    invert_strings: bool = False
    verbose: bool = False
    lyrics: bool = True
    lyrics_model: str = DEFAULT_LYRICS_MODEL
    lyrics_language: str = ""


app = FastAPI(title="TabGrabber Web UI")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (PACKAGE_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/system")
def system_info() -> dict:
    info: dict[str, Any] = {"cuda": False, "gpu": "", "torch": ""}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001 - reported to the page as-is
        info["error"] = str(exc)
    return info


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "audio").name
    if Path(name).suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            400, f"Unsupported audio format. Allowed: "
                 f"{', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}")

    destination = workspace.uploads / f"{uuid.uuid4().hex[:8]}_{safe_name(name)}"
    written = 0
    try:
        with destination.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413, f"File is larger than "
                             f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                out.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return {"path": str(destination), "name": name}


@app.post("/api/jobs")
def create_job(request: JobRequest) -> dict:
    if not request.source.strip() and not request.upload_path.strip():
        raise HTTPException(400, "Give a link or upload a file")
    if not request.formats:
        raise HTTPException(400, "Pick at least one output format")
    if not request.instruments:
        raise HTTPException(400, "Pick at least one instrument")

    input_path = None
    if request.upload_path.strip():
        candidate = Path(request.upload_path).resolve()
        if not candidate.is_relative_to(workspace.uploads.resolve()):
            raise HTTPException(400, "Invalid upload path")
        input_path = candidate

    job = Job(id=uuid.uuid4().hex[:12])
    if input_path is not None:
        job.title = input_path.stem
    _register(job)

    threading.Thread(target=run_job, args=(job, request, input_path),
                     daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _get_job(job_id)
    return {
        "id": job.id,
        "status": job.status,
        "title": job.title,
        "error": job.error,
        "audio": f"/api/audio/{job.id}" if job.source_audio else "",
        "chordsheet": job.sheet_rel,
        "files": collect_files(job) if job.status == "done" else [],
    }


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    job = _get_job(job_id)

    def stream() -> Iterator[str]:
        for line in job.logs:
            yield f"data: {json.dumps({'kind': 'log', 'line': line})}\n\n"
        while True:
            try:
                message = job.events.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(message, ensure_ascii=False)}\n\n"
            if message.get("kind") == "eof":
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/audio/{job_id}")
def job_audio(job_id: str) -> FileResponse:
    """The source audio for a job, streamed so the player can seek."""
    job = _get_job(job_id)
    if not job.source_audio or not job.source_audio.exists():
        raise HTTPException(404, "No audio for this job")
    return FileResponse(job.source_audio)


def _resolve_output(relative: str) -> Path:
    target = (workspace.output / relative).resolve()
    if not target.is_relative_to(workspace.output.resolve()) or not target.is_file():
        raise HTTPException(404, "File not found")
    return target


@app.get("/api/file")
def get_file(path: str, download: bool = False) -> FileResponse:
    target = _resolve_output(path)
    if not download:
        return FileResponse(target)
    return FileResponse(target, filename=target.name,
                        media_type="application/octet-stream")


@app.get("/api/preview")
def preview_text(path: str) -> dict:
    target = _resolve_output(path)
    if target.suffix != ".txt":
        raise HTTPException(400, "Only text tabs can be previewed")
    return {"text": target.read_text(encoding="utf-8", errors="replace")}


# -------------------------------------------------------------------- launch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tabgrabber-web",
        description="Run the TabGrabber web UI on this machine.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"Where downloads and results are stored "
                             f"(default: ./{DEFAULT_DATA_DIR})")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a browser window on startup")
    return parser


def main(argv: list[str] | None = None) -> None:
    global workspace

    try:
        import uvicorn
    except ImportError:
        raise SystemExit(
            "The web UI needs the 'web' extra: pip install 'tabgrabber[web]'")

    args = build_arg_parser().parse_args(argv)
    workspace = Workspace(
        (args.data_dir or Path.cwd() / DEFAULT_DATA_DIR).resolve()).create()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"\n  TabGrabber web UI   {url}")
    print(f"  Data directory      {workspace.root}")
    print("  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
