from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import datetime

VIDEO_DIR = r"D:\veFox"
SERVER_URL = "http://localhost:5557"

app = FastAPI()

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- статика ---
app.mount("/static", StaticFiles(directory=VIDEO_DIR), name="static")

# --- функции ---
# проверка что это действительно канал с папками видео
def check_channel(channel_path):
    full_path = os.path.join(VIDEO_DIR, channel_path)

    videos = os.listdir(full_path)

    if not videos:
        return False

    if os.path.isdir(full_path):
        check_all_videos = any(
            os.path.isdir(os.path.join(full_path, video))
            for video in videos
        )
        return check_all_videos
    else:
        return False

# получаем каналы
def get_channels():
    channels = []

    for channel in os.listdir(VIDEO_DIR):
        if check_channel(channel):
            channels.append({
                'name': channel,
                'avatar': get_channel_avatar(channel)
            })

    return channels

# получаем информацию о видео
def get_video_info(video_path, video_name, channel_name, channel_avatar):
    video_files = os.listdir(video_path)

    if not video_files:
        return None

    video_file = next((f for f in video_files if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))), None)
    preview_file = next((f for f in video_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))), None)

    if video_file:
        full_vide_path = os.path.join(video_path, video_file)
        stats = os.stat(full_vide_path)
        created_at = datetime.datetime.fromtimestamp(stats.st_ctime)

        return {
            'name': video_name,
            'video': f"{SERVER_URL}/static/{channel_name}/{video_name}/{video_file}",
            'preview': f"{SERVER_URL}/static/{channel_name}/{video_name}/{preview_file}" if preview_file else None,
            'channel': channel_name,
            'avatar': channel_avatar,
            'date': created_at.isoformat()
        }

# получаем видео с канала
def get_video_from_channel(channel):
    channel_name = channel["name"]
    channel_avatar = channel["avatar"]

    videos_info = []

    channel_path = os.path.join(VIDEO_DIR, channel_name)

    if not check_channel(channel_path):
        return []

    videos = os.listdir(channel_path)

    if not videos:
        return []

    for video in videos:
        video_path = os.path.join(channel_path, video)

        if not os.path.isdir(video_path):
            continue

        video_files = get_video_info(video_path, video, channel_name, channel_avatar)

        if video_files:
            videos_info.append(video_files)

    return videos_info

# получаем аватарку канала
def get_channel_avatar(channel_name):
    channel_path = os.path.join(VIDEO_DIR, channel_name)
    files = os.listdir(channel_path)

    avatar = next(
        (f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))
        , None
    )

    return f"{SERVER_URL}/static/{channel_name}/{avatar}" if avatar else None

# получаем все видео
def get_all_videos():
    all_videos = []

    channels = get_channels()

    if not channels:
        return []

    for channel in channels:
        videos_info = get_video_from_channel(channel)
        all_videos.extend(videos_info)

    return all_videos


# --- эндпоинты ---
@app.get("/videos")
async def videos():
    return JSONResponse(get_all_videos())
