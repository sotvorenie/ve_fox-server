from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select
import json

from utils import (_safe_listdir, get_tags_by_title, get_create_date,
                   get_video_file, get_photo_file, get_video_url,
                   get_photo_url, get_subtitles_file, get_video_duration,
                   get_subtitles_url, get_and_set_video_info)
from config import VIDEO_DIRECTORY, BASE_STORAGE_DIR, FILMS_DIRECTORY
from database_models import (Channel, Video, Film,
                             FilmGenre, FilmActor)
from database import SessionLocal

from logger import get_logger
logger = get_logger(__name__)


# получаем информацию о канале: дату создания,теги (здесь же создаем (если не создан) info.json)
def get_and_set_channel_info(channel: Path):
    info_file = channel / 'info.json'

    tags = []
    date = ''
    needs_write = False

    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as file:
                info = json.load(file)
                date = info.get('date', '')
                tags = info.get('tags', [])
        except Exception as e:
            logger.warning(f"Не удалось прочитать информацию о канале {channel.name}: {e}")
            needs_write = True
    else:
        needs_write = True

    if not tags:
        tags = get_tags_by_title(channel.name)
        needs_write = True

    if not date:
        date = get_create_date(channel)
        needs_write = True

    if needs_write:
        try:
            with open(info_file, 'w', encoding='utf-8') as file:
                json.dump({'date': date, 'tags': tags}, file, ensure_ascii=False)
            logger.info(f"Обновлена информация в json для канала {channel.name}")
        except Exception as e:
            logger.warning(f"Не удалось записать информацию о канале {channel.name}: {e}")

    return tags, date


# получаем информацию о фильме: дату создания, актеры, жанр (здесь же создаем (если не создан) info.json)
def get_and_set_film_info(film: Path):
    info_file = film / 'info.json'

    actors = []
    genre = ''
    date = ''
    needs_write = False

    if info_file.exists():
        try:
            with open(info_file, 'r', encoding='utf-8') as file:
                info = json.load(file)
                date = info.get('date', '')
                actors = info.get('actors', [])
                genre = info.get('genre', '')
        except Exception as e:
            logger.warning(f"Не удалось прочитать информацию о фильме {film.name}: {e}")
            needs_write = True
    else:
        needs_write = True

    if not date:
        date = get_create_date(film)
        needs_write = True

    if needs_write:
        try:
            with open(info_file, 'w', encoding='utf-8') as file:
                json.dump({'date': date, 'actors': actors, 'genre': genre}, file, ensure_ascii=False)
            logger.info(f"Обновлена информация в json для фильма {film.name}")
        except Exception as e:
            logger.warning(f"Не удалось записать информацию о фильме {film.name}: {e}")

    return actors, genre, date


class SyncState:
    def __init__(self):
        self.existing_channels = {}
        self.existing_videos = {}
        self.existing_films = {}
        self.existing_genres = {}
        self.existing_actors = {}
        self.actual_channels_paths = set()
        self.actual_videos_paths = set()
        self.actual_films_paths = set()


class DateSynchronizer:
    def __init__(self, db: Session):
        self.db = db
        self.state = SyncState()

    def load_existing(self):
        self.state.existing_channels = {
            channel.path: channel
            for channel in self.db.execute(select(Channel)).scalars().all()
        }
        self.state.existing_videos = {
            video.path: video
            for video in self.db.execute(select(Video)).scalars().all()
        }
        self.state.existing_films = {
            film.path: film
            for film in self.db.execute(select(Film)).scalars().all()
        }
        self.state.existing_genres = {
            genre.name: genre
            for genre in self.db.execute(select(FilmGenre)).scalars().all()
        }
        self.state.existing_actors = {
            actor.name: actor
            for actor in self.db.execute(select(FilmActor)).scalars().all()
        }
        self.state.actual_channels_paths = set()
        self.state.actual_videos_paths = set()
        self.state.actual_films_paths = set()

    def delete_empty_video(self, video_duration: int, video_path_str: str, existing_type: str) -> bool:
        existing_obj = self.state.existing_videos if existing_type == 'video' else self.state.existing_films

        if not video_duration or video_duration <= 0:
            if video_path_str in existing_obj:
                self.db.delete(existing_obj[video_path_str])
                del existing_obj[video_path_str]
            logger.warning(f"У видеоматериала {Path(video_path_str).name} duration = 0")
            return True
        return False

    def sync_channels(self):
        for channel_dir in _safe_listdir(VIDEO_DIRECTORY):
            if not channel_dir.is_dir():
                logger.warning(f"Канал {channel_dir.name} не является папкой")
                continue

            tags, date = get_and_set_channel_info(channel_dir)

            channel_avatar = get_photo_file(channel_dir)
            channel_avatar_url = get_photo_url(channel_avatar, channel_dir)

            channel_path_str = channel_dir.relative_to(BASE_STORAGE_DIR).as_posix()
            self.state.actual_channels_paths.add(channel_path_str)

            if channel_path_str in self.state.existing_channels:
                db_channel = self.state.existing_channels[channel_path_str]
                db_channel.name = channel_dir.name
                db_channel.tags = tags
                db_channel.date = date
                db_channel.avatar_url = channel_avatar_url

                channel_id = db_channel.id

                logger.info(f"Обновлена информация в БД для канала {channel_dir.name}")
            else:
                new_channel = Channel(
                    name=channel_dir.name,
                    path=channel_path_str,
                    tags=tags,
                    date=date,
                    avatar_url=channel_avatar_url,
                )
                self.db.add(new_channel)
                self.db.flush()
                self.state.existing_channels[channel_path_str] = new_channel

                channel_id = new_channel.id

                logger.info(f"В БД добавлен новый канал - {channel_dir.name}")

            self.sync_channel_videos(
                channel_dir,
                channel_id,
            )

    def sync_channel_videos(self, channel_path: Path, channel_id: int):
        if not channel_path.is_dir():
            logger.warning(f"Канал {channel_path.name} не является папкой.. Чтение видео невозможно")
            return None

        for video in _safe_listdir(channel_path):
            if not video.is_dir():
                continue

            video_file = get_video_file(video)

            if not video_file:
                logger.warning(f"В папке видео {video.name} нет видеоматериала..")
                continue

            video_path_str = video.relative_to(BASE_STORAGE_DIR).as_posix()

            video_date, video_duration, video_tags = get_and_set_video_info(video, video_file)

            was_deleted_from_db = self.delete_empty_video(video_duration, video_path_str, 'video')
            if was_deleted_from_db:
                continue

            self.state.actual_videos_paths.add(video_path_str)

            video_preview_file = get_photo_file(video)
            video_preview_url = get_photo_url(video_preview_file, video)

            video_url = get_video_url(video_file)

            subtitles_file = get_subtitles_file(video)
            subtitles_url = None
            if subtitles_file:
                subtitles_url = get_subtitles_url(subtitles_file)

            if video_path_str in self.state.existing_videos:
                db_video = self.state.existing_videos[video_path_str]
                db_video.name = video.name
                db_video.date = video_date
                db_video.duration = video_duration
                db_video.tags = video_tags
                db_video.subtitles_url = subtitles_url
                db_video.preview_url = video_preview_url

                logger.info(f"Обновлена информация в БД для видео {video.name}")
            else:
                new_video = Video(
                    name=video.name,
                    path=video_path_str,
                    tags=video_tags,
                    date=video_date,
                    duration=video_duration,
                    preview_url=video_preview_url,
                    video_url=video_url,
                    subtitle_url=subtitles_url,
                    likes=0,
                    views=0,
                    channel_id=channel_id
                )
                self.db.add(new_video)
                self.state.existing_videos[video_path_str] = new_video

                logger.info(f"В БД добавлено новое видео - {video.name}")

    def sync_films(self):
        for film in _safe_listdir(FILMS_DIRECTORY):
            if not film.is_dir():
                logger.warning(f"Фильм {film.name} не является папкой..")
                continue

            film_path_str = str(film)

            film_video_file = get_video_file(film)

            if not film_video_file:
                logger.warning(f"В папке фильма {film.name} нет видеоматериала..")
                continue

            film_actors, film_genre, film_date = get_and_set_film_info(film)

            film_duration = get_video_duration(film_video_file)

            was_deleted_from_db = self.delete_empty_video(film_duration, film_path_str, 'film')
            if was_deleted_from_db:
                continue

            self.state.actual_films_paths.add(film_path_str)

            preview_file = get_photo_file(film)
            preview_url = get_photo_url(preview_file, film)

            video_url = get_video_url(film_video_file)

            if film_genre not in self.state.existing_genres:
                new_genre = FilmGenre(name=film_genre)
                self.db.add(new_genre)
                self.db.flush()
                self.state.existing_genres[film_genre] = new_genre

                logger.info(f"В БД добавлен новый жанр - {film_genre}")

            db_genre = self.state.existing_genres[film_genre]

            db_actors = []
            for actor in film_actors:
                if actor not in self.state.existing_actors:
                    new_actor = FilmActor(name=actor)
                    self.db.add(new_actor)
                    self.db.flush()
                    self.state.existing_actors[actor] = new_actor

                    logger.info(f"В БД добавлен новый актер - {actor}")

                db_actors.append(self.state.existing_actors[actor])

            if film_path_str in self.state.existing_films:
                db_film = self.state.existing_films[film_path_str]
                db_film.name = film.name
                db_film.date = film_date
                db_film.duration = film_duration
                db_film.preview_url = preview_url
                db_film.actors = db_actors
                db_film.genre = db_genre

                logger.info(f"Обновлена информация в БД для фильма {film.name}")
            else:
                new_film = Film(
                    name=film.name,
                    path=film_path_str,
                    video_url=video_url,
                    duration=film_duration,
                    preview_url=preview_url,
                )
                new_film.genre = db_genre
                new_film.actors = db_actors
                self.db.add(new_film)
                self.state.existing_films[film_path_str] = new_film

                logger.info(f"В БД добавлен новый фильм - {film.name}")

    def delete_unused_from_db(self):
        channels_paths_in_db = set(self.state.existing_channels.keys())
        channels_path_to_delete = channels_paths_in_db - self.state.actual_channels_paths
        for channel_path in channels_path_to_delete:
            self.db.delete(self.state.existing_channels[channel_path])

        videos_paths_in_db = set(self.state.existing_videos.keys())
        videos_path_to_delete = videos_paths_in_db - self.state.actual_videos_paths
        for video_path in videos_path_to_delete:
            self.db.delete(self.state.existing_videos[video_path])

        films_paths_in_db = set(self.state.existing_films.keys())
        films_path_to_delete = films_paths_in_db - self.state.actual_films_paths

        for film_path in films_path_to_delete:
            self.db.delete(self.state.existing_films[film_path])

    def sync(self):
        self.load_existing()
        self.sync_channels()
        self.sync_films()
        self.delete_unused_from_db()


# проходимся по всем видео/фильмам и записываем их данные в бд
def start_db():
    try:
        with SessionLocal() as db:
            logger.info("Начало синхронизации..")
            with db.begin():
                synchronizer = DateSynchronizer(db)
                synchronizer.sync()
            logger.info("Синхронизация данных прошла успешно!!")

    except Exception as e:
        logger.critical(f"Ошибка при наполнении БД видео: {e}")
