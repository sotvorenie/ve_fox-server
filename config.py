from pathlib import Path

BASE_STORAGE_DIR = Path(r"E:\server\veFox").resolve()

VIDEO_DIRECTORY = BASE_STORAGE_DIR / "videos"
FILMS_DIRECTORY = BASE_STORAGE_DIR / "films"
AVATARS_DIRECTORY = BASE_STORAGE_DIR / "users_avatars"

ALLOWED_VIDEO_SUFFIX = {".mp4", ".mkv", ".avi", ".mov"}
ALLOWED_SUBTITLES_SUFFIX = {".vtt"}
ALLOWED_PHOTO_SUFFIX = {".jpg", ".jpeg", ".png", ".webp"}
