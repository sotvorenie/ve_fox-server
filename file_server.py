import random
import re
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import datetime
import logging
from typing import List, Optional, Tuple

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
    video_path: str
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

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def _safe_listdir(path: Path):
    try:
        return list(path.iterdir())
    except Exception:
        return []

def normalize(s: str) -> str:
    """Нормализация строки для сравнения."""
    if not s:
        return ""
    s = s.strip().lower()
    s = s.replace("►", " ").replace("–", " ").replace("—", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def extract_base(name: str) -> str:
    """Берём базовое название игры/сериала до первой стрелки."""
    name = normalize(name)
    if " ► " in name:
        return name.split(" ► ")[0]
    return name

def parse_series(name: str) -> Optional[Tuple[str, Optional[int], int]]:
    """
    Возвращает (base_name, season_or_None, episode)
    Для большинства летсплеев и серий.
    """
    s = normalize(name)
    m = re.search(r"(?:№|#|серия|часть|ep|эп)\s*\.?\s*(\d+)", s)
    if not m:
        return None
    episode = int(m.group(1))
    base = extract_base(s)
    return base, None, episode

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
            "video_path": f"{channel_name}/{video_folder.name}",
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
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": videos[start:end]
    }

@app.get("/video", response_model=VideoInfo)
def get_video(video_path: str):
    full_path = VIDEO_DIR / video_path

    if not full_path.exists() or not full_path.is_dir():
        raise HTTPException(status_code=404, detail="Video not found")

    files = _safe_listdir(full_path)

    video_file = next((f.name for f in files if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_EXTS), None)

    if not video_file:
        raise HTTPException(status_code=404, detail="Video not found")

    channel_name = full_path.parent.name
    channel = next((c for c in _cache["channels"] if c["name"] == channel_name), None)
    channel_avatar = channel["avatar"] if channel else None

    try:
        stats = (full_path / video_file).stat()
        created_at = datetime.datetime.fromtimestamp(getattr(stats, "st_ctime", stats.st_mtime)).isoformat()
    except Exception:
        created_at = datetime.datetime.now().isoformat()

    video_url = f"{SERVER_URL}/static/{video_path}/{video_file}"

    return JSONResponse({
        "name": full_path.name,
        "video": video_url,
        "preview": "",
        "channel": channel_name,
        "avatar": channel_avatar,
        "date": created_at
    })

@app.get("/channel/{channel_name}/videos", response_model=ResponseData)
def channel_videos(channel_name: str, page: int = 1, is_new: bool = True, limit: int = 20):
    channel = next((c for c in _cache["channels"] if c["name"] == channel_name), None)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    videos = [v for v in _cache["videos"] if normalize(v["channel"]) == normalize(channel_name)]
    videos.sort(key=lambda v: v["date"], reverse=is_new)
    total = len(videos)
    start = max((page - 1) * limit, 0)
    end = start + limit
    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": videos[start:end]
    }

@app.get("/search", response_model=ResponseData)
def search(name: str, page: int = 1, limit: int = 20):
    q = normalize(name)
    if not q:
        return {"total": 0, "page": page, "limit": limit, "has_more": False, "videos": []}
    videos = [v for v in _cache["videos"] if q in normalize(v["name"]) or q in normalize(v["channel"])]
    total = len(videos)
    start = max((page - 1) * limit, 0)
    end = start + limit
    return {"total": total, "page": page, "limit": limit, "has_more": end < total, "videos": videos[start:end]}

@app.get("/recommended", response_model=ResponseData)
def get_recommendations(name: str, channel_name: str, page: int = 1, limit: int = 20):
    videos = _cache["videos"]
    weights = defaultdict(int)

    parsed = parse_series(name)
    if parsed:
        base, _, episode = parsed
        for v in videos:
            p = parse_series(v["name"])
            if not p:
                continue
            v_base, _, v_episode = p
            if v_base == base:
                if episode is not None:
                    if v_episode == episode + 1:
                        weights[v["video_path"]] += 1000
                    elif v_episode == episode + 2:
                        weights[v["video_path"]] += 800

    # Видео с того же канала
    if channel_name:
        for v in videos:
            if normalize(v["channel"]) == normalize(channel_name):
                weights[v["video_path"]] += 500

    # Похожие по словам
    query_words = set(normalize(name).split())
    for v in videos:
        for w in query_words:
            if w in normalize(v["name"]):
                weights[v["video_path"]] += 10

    # Рандомный вес
    for v in videos:
        weights[v["video_path"]] += random.randint(0, 5)

    # Сортировка
    sorted_videos = sorted(videos, key=lambda v: weights[v["video_path"]], reverse=True)

    # Пагинация
    total = len(sorted_videos)
    start = max((page - 1) * limit, 0)
    end = start + limit

    return {"total": total, "page": page, "limit": limit, "has_more": end < total, "videos": sorted_videos[start:end]}

# --- ОБРАБОТКА ОШИБОК ---
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)