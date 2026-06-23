import math
import re
from pathlib import Path
import functools
from datetime import datetime, timezone
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import exists, and_, func, select
from fastapi import Depends, HTTPException
import cv2
import random
import json

from config import ALLOWED_VIDEO_SUFFIX, ALLOWED_PHOTO_SUFFIX, VIDEO_DIRECTORY, ALLOWED_SUBTITLES_SUFFIX
from auth import get_db
from database_models import WatchLater, Video
from httpExceptions import db_exception

from logger import get_logger

logger = get_logger(__name__)


# безопасное открытие папки
def _safe_listdir(path: Path):
    try:
        return list(path.iterdir())
    except Exception as e:
        logger.error(f"Ошибка: файл в {path} не найден. Сообщение ошибки: {e}")
        return []


# убираем из названия видео/канала ненужные символы
def normalize_name(name: str) -> str:
    if not name:
        return ""

    return ((name.strip().lower()
             .replace("►", " ")
             .replace("–", " ")
             .replace("—", " "))
            .replace("!", " ")
            .replace("!!", " "))


# убираем из названия папки для видео ненужные символы
def normalize_path_name(name: str) -> str:
    name = name.strip().lower()

    return re.sub(
        r'[\\/:*?"<>|]+',
        '_',
        name
    )


# если у видео нет превью - получаем его рандомно
def create_video_preview(video_file: Path, output_preview_path: Path):
    cap = cv2.VideoCapture(str(video_file))

    try:
        if not cap.isOpened():
            logger.warning(f"Предупреждение: OpenCV не смог открыть {video_file.name}")
            return False

        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.get(cv2.CAP_PROP_FPS)

        if frames <= 0:
            logger.warning(f"Ошибка: В видео {video_file.name} 0 кадров")
            return False

        random_frame = random.randint(int(frames * 0.1), int(frames * 0.9)) if frames > 10 else 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)

        success, frame = cap.read()
        if success and frame is not None:
            height, width = frame.shape[:2]
            new_width = 1280
            new_height = int(height * (new_width / width))
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

            is_success, buffer = cv2.imencode(".jpg", resized_frame)
            if is_success:
                with open(output_preview_path, "wb") as f:
                    f.write(buffer.tobytes())
                logger.info(f"Создано превью для {video_file.name}")
                return True
    except Exception as e:
        logger.warning(f"Ошибка при создании превью для {video_file.name}: ", e)
    finally:
        cap.release()
    return False


# получаем теги по названию канала/видео
def get_tags_by_title(title: str) -> list[str]:
    normalize_video_name = normalize_name(title)

    if not normalize_video_name:
        return []

    return normalize_video_name.split()


# получаем длительность видео
def get_video_duration(video_file: Path):
    cap = cv2.VideoCapture(str(video_file))

    try:
        if not cap.isOpened():
            return 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps <= 0 or frames <= 0:
            return 0

        return math.ceil(frames / fps)
    except Exception as e:
        logger.warning(f"Не удалось прочитать видео {video_file.name}: ", e)
        return 0
    finally:
        cap.release()


# получаем информацию о видео: дату создания, длительность, теги (здесь же создаем (если не создан) info.json)
def get_and_set_video_info(video: Path, video_file: Path):
    info_file = video / "info.json"

    tags = []
    needs_write = False

    if info_file.exists():
        try:
            with open(info_file, "r", encoding='utf-8') as file:
                data = json.load(file)
                tags = data.get('tags', [])
        except Exception as e:
            logger.warning(f"Не удалось прочитать видео {video_file.name}: ", e)
            needs_write = True
    else:
        needs_write = True

    date = get_create_date(video_file)
    duration = get_video_duration(video_file)

    if not tags:
        tags = get_tags_by_title(video.name)
        needs_write = True

    if needs_write:
        try:
            with open(info_file, "w", encoding='utf-8') as file:
                json.dump({
                    'tags': tags
                }, file, ensure_ascii=False)
            logger.info(f"Обновлена информация в json для видео {video_file.name}")
        except Exception as e:
            logger.warning(f"Ошибка записи информации о видео {video_file.name}: ", e)

    return date, duration, tags


# получаем дату создания видео/канала
def get_create_date(file: Path) -> datetime:
    try:
        mtime = file.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc)
    except Exception as e:
        logger.warning(f"Ошибка получения даты для {file.name}: ", e)
        return datetime.now(timezone.utc)


# получаем видео-файл
def get_video_file(path: Path):
    return next(
        (f for f in _safe_listdir(path) if f.is_file() and f.suffix.lower() in ALLOWED_VIDEO_SUFFIX), None
    )


# получаем url видео-файла
def get_video_url(file: Path):
    return f"/static/{file.relative_to(VIDEO_DIRECTORY).as_posix()}"


# получаем фото-файл
def get_photo_file(path: Path):
    return next(
        (f for f in _safe_listdir(path) if f.is_file() and f.suffix.lower() in ALLOWED_PHOTO_SUFFIX), None
    )


# получаем url фото-файла
def get_photo_url(file: Path, path: Path):
    photo_url = None
    if file:
        photo_url = f"/static/{file.relative_to(VIDEO_DIRECTORY).as_posix()}"
    else:
        preview_path = path / "превью.jpg"
        if create_video_preview(file, preview_path):
            photo_url = f"/static/{preview_path.relative_to(VIDEO_DIRECTORY).as_posix()}"

    return photo_url


# получаем файл субтитров
def get_subtitles_file(video: Path):
    return next(
        (f for f in _safe_listdir(video) if
         f.is_file() and f.suffix.lower() in ALLOWED_SUBTITLES_SUFFIX), None
    )


# получаем url субтитров
def get_subtitles_url(subtitles: Path):
    return f"/static/{subtitles.relative_to(VIDEO_DIRECTORY).as_posix()}"


# проверяем: добавлено ли видео в раздел "Смотреть позже"
def check_watch_later(video_id, user_id, db: Session = Depends(get_db)):
    return db.scalar(select(exists().where(and_(WatchLater.video_id == video_id, WatchLater.user_id == user_id))))


# получаем offset
def get_offset(page: int = 1, limit: int = 21):
    return (page - 1) * limit


# декоратор для запросов с бд
def db_transaction(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        db = next((v for v in kwargs.values() if isinstance(v, Session)), None)
        try:
            result = function(*args, **kwargs)
            if db:
                db.commit()
            return result
        except HTTPException:
            if db:
                db.rollback()
            raise
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Ошибка базы данных: ", e)
            raise db_exception

    return wrapper


# получаем путь файла
def get_file_path(url: str):
    if not url:
        return None
    relative_path = Path(url.replace("static/", "", 1))
    return VIDEO_DIRECTORY / relative_path


# получаем url файла
def get_file_url(path: Path):
    if not path:
        return None
    return f"static/{path.relative_to(VIDEO_DIRECTORY).as_posix()}"


# получаем total и список видео с таблицы: для watch_later, like, history
def get_total_and_videos_from_db(model, user_id, page, limit, db):
    skip = get_offset(page, limit)

    videos = db.scalars(select(model)
                        .where(model.user_id == user_id)
                        .options(joinedload(model.video).joinedload(Video.channel))
                        .order_by(model.date.desc())
                        .offset(skip)
                        .limit(limit)).all()
    total = db.scalar(select(func.count(model.id)).where(model.user_id == user_id))

    return {
        "videos": videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total,
    }
