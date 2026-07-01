import datetime
import uuid
from fastapi import (APIRouter, Depends, UploadFile,
                     Form, File, HTTPException)
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from pathlib import Path
import shutil

from models import VideoResponse
from database_models import (User, Channel, ChannelSection,
                             Video)
from httpExceptions import (channel_exception, section_exception, uploaded_video_exception,
                            video_format_exception, photo_format_exception)
from auth import get_user
from config import (VIDEO_DIRECTORY, ALLOWED_VIDEO_SUFFIX, ALLOWED_PHOTO_SUFFIX,
                    BASE_STORAGE_DIR)
from utils import (normalize_path_name, get_file_url, create_video_preview,
                   get_tags_by_title, db_transaction, get_and_set_video_info)
from database import get_db

from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])


def get_channel_and_check_section(channel_id: int, section_id: int | None, db: Session):
    channel = db.get(Channel, channel_id)
    if not channel:
        raise channel_exception

    if section_id:
        section = db.get(ChannelSection, section_id)
        if not section or section.channel_id != channel_id:
            raise section_exception

    return channel


def get_video(video: UploadFile, video_directory: Path, tags: list):
    video_suffix = Path(video.filename).suffix.lower()
    if video_suffix not in ALLOWED_VIDEO_SUFFIX:
        raise video_format_exception
    video_file_path = video_directory / f"video{video_suffix}"
    video_url = get_file_url(video_file_path)
    try:
        with video_file_path.open("wb") as f:
            shutil.copyfileobj(video.file, f)
    finally:
        video.file.close()
    _, video_duration, video_tags = get_and_set_video_info(video_directory, video_file_path, tags)
    if not video_duration:
        raise uploaded_video_exception

    return video_duration, video_url, video_file_path, video_tags


def get_section_index(section_id: int, db: Session):
    section_index = 1
    if section_id:
        last_section_video_index = db.scalar(
            select(func.max(Video.section_index))
            .where(Video.section_id == section_id)
        )
        section_index = (last_section_video_index or 0) + 1

    return section_index


def get_preview(preview: UploadFile, video_directory: Path, video_file_path: Path):
    preview_suffix = ".jpg"
    if preview:
        preview_suffix = Path(preview.filename).suffix.lower()
        if preview_suffix not in ALLOWED_PHOTO_SUFFIX:
            raise photo_format_exception
    preview_file_path = video_directory / f"preview{preview_suffix}"
    preview_url = None
    if preview:
        try:
            with preview_file_path.open("wb") as f:
                shutil.copyfileobj(preview.file, f)
        finally:
            preview.file.close()
        preview_url = get_file_url(preview_file_path)
    else:
        preview = create_video_preview(video_file_path, preview_file_path)
        if preview:
            preview_url = get_file_url(preview_file_path)

    return preview_url


@router.post("/video", response_model=VideoResponse)
@db_transaction
def upload_video(
        video: UploadFile = File(...),
        preview: UploadFile | None = File(None),
        title: str = Form(...),
        tags: list[str] | None = Form(None),
        channel_id: int = Form(...),
        section_id: int | None = Form(None),
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    video_directory = None

    try:
        channel = get_channel_and_check_section(channel_id, section_id, db)

        video_folder_name = normalize_path_name(title)
        video_directory = VIDEO_DIRECTORY / channel.name / video_folder_name
        video_directory.mkdir(parents=True, exist_ok=True)

        video_duration, video_url, video_file_path, video_tags = get_video(video, video_directory, tags)

        preview_url = get_preview(preview, video_directory, video_file_path)

        section_index = get_section_index(section_id, db)

        tags = tags or get_tags_by_title(title)

        new_video = Video(
            name=video_folder_name,
            path=video_directory.relative_to(BASE_STORAGE_DIR).as_posix(),
            tags=list(set(video_tags + tags)),
            date=datetime.datetime.now(datetime.timezone.utc),
            duration=video_duration,
            preview_url=preview_url,
            video_url=video_url,
            subtitle_url=None,
            likes=0,
            views=0,
            channel_id=channel_id,
            section_id=section_id,
            section_index=section_index
        )
        db.add(new_video)

        return new_video

    except HTTPException:
        if video_directory and video_directory.exists():
            shutil.rmtree(video_directory, ignore_errors=True)
        raise

    except Exception:
        logger.exception("Ошибка загрузки видео")

        if video_directory and video_directory.exists():
            shutil.rmtree(video_directory, ignore_errors=True)

        raise
