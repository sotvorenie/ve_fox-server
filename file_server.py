from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import os
import time
import datetime
import asyncio
import logging
from typing import List, Optional
import random

# --- НАСТРОЙКИ --- #
VIDEO_DIR = Path(r"D:\veFox")
SERVER_URL = "http://localhost:5557"
CACHE_TTL = 8  # время жизни кэша
ALLOWED_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}
ALLOWED_PREVIEW_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# ----------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("video_server")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=600,
)

if VIDEO_DIR.exists() and VIDEO_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(VIDEO_DIR)), name="static")
else:
    logger.warning(f"VIDEO_DIR {VIDEO_DIR} не найден — статические файлы не будут смонтированы")

# ----------------- МОДЕЛИ -----------------
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

# ----------------- Cache & Scanner -----------------
_cache = {
    "videos": [],  # список VideoInfo dict
    "channels": [],
    "updated_at": 0.0,
}
_cache_lock = asyncio.Lock()

async def refresh_cache(force: bool = False):
    async with _cache_lock:
        now = time.time()
        if not force and now - _cache["updated_at"] < CACHE_TTL:
            logger.debug("Кеш актуален — обновление пропущено")
            return

        logger.info("Обновляю кеш видео...")
        try:
            channels = _scan_channels()
            videos = []
            for ch in channels:
                vids = _scan_videos_in_channel(ch["name"], ch["avatar"])
                videos.extend(vids)

            _cache["channels"] = channels
            _cache["videos"] = videos
            _cache["updated_at"] = now
            logger.info(f"Кеш обновлён: {len(channels)} каналов, {len(videos)} видео")
        except Exception as e:
            logger.exception("Ошибка при обновлении кеша")

# --- ФУНКЦИИ ---
def _safe_listdir(path: Path):
    try:
        return list(path.iterdir())
    except Exception:
        logger.debug(f"Не удалось прочитать папку {path}")
        return []


def _scan_channels():
    res = []
    if not VIDEO_DIR.exists() or not VIDEO_DIR.is_dir():
        return res

    try:
        for entry in VIDEO_DIR.iterdir():
            if not entry.is_dir():
                continue

            try:
                children = [c for c in entry.iterdir() if c.is_dir()]
            except Exception:
                children = []

            if not children:
                continue

            avatar = None
            try:
                for f in entry.iterdir():
                    if f.is_file() and f.suffix.lower() in ALLOWED_PREVIEW_EXTS:
                        avatar = f.name
                        break
            except Exception:
                avatar = None

            res.append({
                "name": entry.name,
                "avatar": f"{SERVER_URL}/static/{entry.name}/{avatar}" if avatar else None,
            })
    except Exception:
        logger.exception("Ошибка при сканировании каналов")

    return res


def _scan_videos_in_channel(channel_name: str, channel_avatar: Optional[str]):
    res = []
    channel_path = VIDEO_DIR / channel_name
    if not channel_path.exists() or not channel_path.is_dir():
        return res

    try:
        for video_folder in channel_path.iterdir():
            if not video_folder.is_dir():
                continue

            try:
                files = list(video_folder.iterdir())
            except Exception:
                logger.debug(f"Не удалось прочитать видео-папку {video_folder}")
                continue

            video_file = None
            preview_file = None

            for f in files:
                if not f.is_file():
                    continue
                sfx = f.suffix.lower()
                if not video_file and sfx in ALLOWED_VIDEO_EXTS:
                    video_file = f.name
                if not preview_file and sfx in ALLOWED_PREVIEW_EXTS:
                    preview_file = f.name
                if video_file and preview_file:
                    break

            if not video_file:
                continue

            try:
                stats = video_folder.joinpath(video_file).stat()
                created_at = datetime.datetime.fromtimestamp(getattr(stats, 'st_ctime', stats.st_mtime)).isoformat()
            except Exception:
                created_at = datetime.datetime.now().isoformat()

            res.append({
                "name": video_folder.name,
                "video": f"{SERVER_URL}/static/{channel_name}/{video_folder.name}/{video_file}",
                "preview": f"{SERVER_URL}/static/{channel_name}/{video_folder.name}/{preview_file}" if preview_file else None,
                "channel": channel_name,
                "avatar": channel_avatar,
                "date": created_at,
            })
    except Exception:
        logger.exception(f"Ошибка при сканировании видео в канале {channel_name}")

    return res

# ----------------- API Эндпоинты -----------------
@app.on_event("startup")
async def startup_event():
    await refresh_cache(force=True)


@app.get("/test")
async def test_endpoint():
    return JSONResponse({
        "message": "Привет! Сервер жив.",
        "cached_at": _cache["updated_at"],
        "videos_cached": len(_cache["videos"]),
    })


@app.post("/refresh_cache")
async def api_refresh_cache():
    """Принудительное обновление кеша."""
    await refresh_cache(force=True)
    return JSONResponse({"status": "ok", "updated_at": _cache["updated_at"]})


@app.get("/channels")
async def api_get_channels():
    await refresh_cache()
    return JSONResponse({"total": len(_cache["channels"]), "channels": _cache["channels"]})


@app.get("/all_videos", response_model=ResponseData)
async def api_all_videos(page: int = 1, limit: int = 20):
    await refresh_cache()
    videos = _cache["videos"].copy()

    total = len(videos)

    random.shuffle(videos)

    start = max((page - 1) * limit, 0)
    end = start + limit
    paginated = videos[start:end]

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": paginated,
    }


@app.get("/channel/{channel_name}/videos", response_model=ResponseData)
async def api_channel_videos(channel_name: str, page: int = 1, limit: int = 20):
    await refresh_cache()

    channels = _cache["channels"]
    channel = next((c for c in channels if c["name"] == channel_name), None)
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
        "videos": paginated,
    }


@app.get("/search")
async def api_search(q: str, page: int = 1, limit: int = 20):
    await refresh_cache()
    ql = q.lower().strip()
    videos = [v for v in _cache["videos"] if ql in v["name"].lower() or ql in v["channel"].lower()]

    total = len(videos)
    start = max((page - 1) * limit, 0)
    end = start + limit
    paginated = videos[start:end]

    return JSONResponse({
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": paginated,
    })

# ----------------- Обработка ошибок -----------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)

# ----------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("optimized_fastapi_video_server:app", host="0.0.0.0", port=5557, reload=True)
