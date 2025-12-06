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
    Возвращает (base_name, season, episode)
    Теперь супер-надёжный парсер.
    """
    s = normalize(name)

    # 1) Ищем стандартные эпизодные маркеры
    patterns = [
        r'(?:№|#)\s*(\d+)',              # №3, #4
        r'(?:серия|эпизод|эп|часть|ч)\s*\.?\s*(\d+)',   # серия 3, эп 2, часть 4
        r'(?:episode|ep)\s*\.?\s*(\d+)',               # episode 3, ep 2
        r'(?:выпуск)\s*(\d+)',                         # выпуск 5
    ]

    episode = None
    for p in patterns:
        m = re.search(p, s, re.IGNORECASE)
        if m:
            episode = int(m.group(1))
            break

    # 2) Если не нашли — ищем одиночные числа вида " ► 3" или " 3 "
    if episode is None:
        m = re.search(r'[^\d](\d{1,3})[^\d]', s)
        if m:
            # Проверка, чтобы не брать год, типо 2021
            num = int(m.group(1))
            if 1 <= num <= 999:
                episode = num

    if episode is None:
        return None

    # Определяем base_name как текст ДО последнего "►"
    if "►" in name:
        base = name.split("►")[0].strip().lower()
    else:
        base = name.strip().lower()

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

    query_words = q.split()

    results = []

    for v in _cache["videos"]:
        title = normalize(v["name"])
        channel = normalize(v["channel"])

        # Если хотя бы одно слово запроса встречается в названии или канале
        match_score = 0

        for word in query_words:
            if len(word) < 2:
                continue

            # Прямое совпадение слова
            if word in title or word in channel:
                match_score += 2
                continue

            # Совпадение по отдельным токенам (например dying + light)
            for part in title.split():
                if part.startswith(word):
                    match_score += 1

        # Парсинг серии — чтобы "2" находило "№2"
        parsed = parse_series(v["name"])
        if parsed:
            _, _, ep = parsed
            if str(ep) in query_words:
                match_score += 3  # серия совпала → очень важно

        if match_score > 0:
            results.append((match_score, v))

    # Сортировка по релевантности
    results.sort(key=lambda x: x[0], reverse=True)
    videos = [v for _, v in results]

    total = len(videos)
    start = (page - 1) * limit
    end = start + limit

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": videos[start:end]
    }

@app.get("/recommended", response_model=ResponseData)
def get_recommendations(name: str, channel_name: str, page: int = 1, limit: int = 20):
    videos = _cache["videos"]
    weights = {}

    # ---------------------------------------------------------
    # 1. Парсинг текущего видео: base_name + episode_number
    # ---------------------------------------------------------
    parsed = parse_series(name)

    base = None
    episode = None

    if parsed:
        base, _, episode = parsed

    # Для сортировки строго эпизоды идут в правильном порядке
    next_ep_candidates = []
    same_channel_candidates = []
    similar_candidates = []
    random_candidates = []

    normalized_name = normalize(name)
    normalized_channel = normalize(channel_name)

    query_words = set(normalized_name.split())

    # ---------------------------------------------------------
    # Сканируем все видео
    # ---------------------------------------------------------
    for v in videos:
        v_name_norm = normalize(v["name"])
        v_channel_norm = normalize(v["channel"])

        if v_name_norm == normalized_name and v_channel_norm == normalized_channel:
            continue  # пропустить текущее видео

        v_parsed = parse_series(v["name"])

        # ----- 1. Следующие эпизоды -----
        if parsed and v_parsed:
            v_base, _, v_episode = v_parsed

            if v_base == base:
                if v_episode == episode + 1:
                    next_ep_candidates.append((1, v))
                    continue
                if v_episode == episode + 2:
                    next_ep_candidates.append((2, v))
                    continue

        # ----- 2. Видео с того же канала -----
        if v_channel_norm == normalized_channel:
            same_channel_candidates.append(v)
            continue

        # ----- 3. Похожие по названию -----
        common = 0
        for w in query_words:
            if len(w) > 2 and w in v_name_norm:
                common += 1

        if common > 0:
            similar_candidates.append((common, v))
            continue

        # ----- 4. Рандомный хвост -----
        random_candidates.append(v)

    # ---------------------------------------------------------
    # Формируем итоговый список
    # ---------------------------------------------------------
    final_list = []

    # 1) Строго упорядочить эпизоды: сначала N+1, потом N+2
    next_ep_sorted = sorted(next_ep_candidates, key=lambda x: x[0])
    final_list.extend([v for _, v in next_ep_sorted])

    # 2) Взять 3–5 видео с того же канала
    same_channel_sorted = same_channel_candidates[:5]
    final_list.extend(same_channel_sorted)

    # 3) Похожие по названию: сортировать по убыванию совпадений
    similar_sorted = sorted(similar_candidates, key=lambda x: x[0], reverse=True)
    final_list.extend([v for _, v in similar_sorted])

    # 4) Хвостовое рандомное перемешивание
    random.shuffle(random_candidates)
    final_list.extend(random_candidates)

    # ---------------------------------------------------------
    # Пагинация
    # ---------------------------------------------------------
    total = len(final_list)
    start = (page - 1) * limit
    end = start + limit

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": end < total,
        "videos": final_list[start:end]
    }

# --- ОБРАБОТКА ОШИБОК ---
@app.exception_handler(Exception)
def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse({"detail": "Internal Server Error"}, status_code=500)