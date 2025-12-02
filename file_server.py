import random

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import datetime
import logging
from typing import List, Optional

# --- НАСТРОЙКИ --- #
VIDEO_DIR = Path(r"D:\veFox")
SERVER_URL = "http://localhost:5557"
ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
ALLOWED_PREVIEW_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# ----------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video_server")

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

# --- Статика ---
if VIDEO_DIR.exists() and VIDEO_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(VIDEO_DIR)), name="static")
else:
    logger.warning(f"VIDEO_DIR {VIDEO_DIR} не найден — статические файлы не будут смонтированы")

# --- МОДЕЛИ ---
class VideoInfo(BaseModel):
    name: str
    video: str
    preview: Optional[str]
    channel: str
    avatar: Optional[str]
    date: str

class ResponseData(BaseModel):
    total: int
    page: int
    limit: int
    has_more: bool
    videos: List[VideoInfo]

# --- КЭШ ---
_cache = {
    "channels": [],
    "videos": [],
}

# --- ФУНКЦИИ ---
def _safe_listdir(path: Path):
    try:
        return list(path.iterdir())
    except Exception:
        return []

def _scan_channels():
    channels = []
    for entry in _safe_listdir(VIDEO_DIR):
        if not entry.is_dir():
            continue

        subfolders = [c for c in _safe_listdir(entry) if c.is_dir()]
        if not subfolders:
            continue

        avatar_file = next(
            (f.name for f in _safe_listdir(entry) if f.is_file() and f.suffix.lower() in ALLOWED_PREVIEW_EXTS),
            None
        )

        channels.append({
            "name": entry.name,
            "avatar": f"{SERVER_URL}/static/{entry.name}/{avatar_file}" if avatar_file else None
        })

    return channels

def _scan_videos_in_channel(channel_name: str, channel_avatar: Optional[str]):
    videos = []
    channel_path = VIDEO_DIR / channel_name
    for video_folder in _safe_listdir(channel_path):
        if not video_folder.is_dir():
            continue

        files = _safe_listdir(video_folder)

        video_file = next((f.name for f in files if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTS), None)
        preview_file = next((f.name for f in files if f.is_file() and f.suffix.lower() in ALLOWED_PREVIEW_EXTS), None)

        if not video_file:
            continue

        try:
            stats = (video_folder / video_file).stat()
            created_at = datetime.datetime.fromtimestamp(getattr(stats, "st_ctime", stats.st_mtime)).isoformat()
        except Exception:
            created_at = datetime.datetime.now().isoformat()

        videos.append({
            "name": video_folder.name,
            "video": f"{SERVER_URL}/static/{channel_name}/{video_folder.name}/{video_file}",
            "preview": f"{SERVER_URL}/static/{channel_name}/{video_folder.name}/{preview_file}" if preview_file else None,
            "channel": channel_name,
            "avatar": channel_avatar,
            "date": created_at
        })

    videos.sort(key=lambda v: v["date"], reverse=True)
    return videos

def build_cache():
    channels = _scan_channels()
    videos = []
    for ch in channels:
        vids = _scan_videos_in_channel(ch["name"], ch["avatar"])
        videos.extend(vids)
    _cache["channels"] = channels
    _cache["videos"] = videos
    logger.info(f"Кэш загружен: {len(channels)} каналов, {len(videos)} видео")

# --- STARTUP ---
@app.on_event("startup")
def startup_event():
    build_cache()

# --- ЭНДПОИНТЫ ---
@app.get("/test")
def test_endpoint():
    return JSONResponse({
        "message": "Привет! Сервер жив.",
        "videos_cached": len(_cache["videos"]),
        "channels_cached": len(_cache["channels"])
    })

@app.get("/channels")
def get_channels():
    return JSONResponse({
        "total": len(_cache["channels"]),
        "channels": _cache["channels"]
    })

@app.get("/all_videos", response_model=ResponseData)
def all_videos(page: int = 1, limit: int = 20):
    videos = _cache["videos"].copy()

    random.shuffle(videos)

    total = len(videos)

    start = max((page - 1) * limit, 0)
    end = start + limit
    paginated = videos[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": paginated
    }

@app.get("/channel/{channel_name}/videos", response_model=ResponseData)
def channel_videos(channel_name: str, page: int = 1, limit: int = 20):
    channel = next((c for c in _cache["channels"] if c["name"] == channel_name), None)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")

    videos = [v for v in _cache["videos"] if v["channel"] == channel_name]

    videos.sort(key=lambda v: v["date"], reverse=True)

    total = len(videos)
    start = max((page - 1) * limit, 0)
    end = start + limit
    paginated = videos[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": paginated
    }

@app.get("/search", response_model=ResponseData)
def search(name: str, page: int = 1, limit: int = 20):
    q = name.lower().strip()
    videos = [v for v in _cache["videos"] if q in v["name"].lower() or q in v["channel"].lower()]

    # videos.sort(key=lambda v: v["date"], reverse=True)

    total = len(videos)
    start = max((page - 1) * limit, 0)
    end = start + limit
    paginated = videos[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": paginated
    }

# --- ОБРАБОТКА ОШИБОК ---
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)