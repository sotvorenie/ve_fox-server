from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func, select, update, and_
from pathlib import Path

from models import VideosListResponse, VideoResponse
from database_models import Video
from database import get_db
from utils import get_video_file, get_offset, db_transaction
from httpExceptions import video_exception

from logger import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/video", tags=["Video"])


@router.get("/all", response_model=VideosListResponse)
@db_transaction
def get_all_videos(page: int = 1, limit: int = 21, seed: float = 0.5, db: Session = Depends(get_db)):
    skip = get_offset(page, limit)

    db.execute(text("SELECT setseed(:s)"), {"s": seed})

    videos = db.scalars(select(Video)
                        .options(joinedload(Video.channel))
                        .order_by(func.random())
                        .offset(skip)
                        .limit(limit)
                        ).all()

    total = db.execute(select(func.count(Video.id))).scalar_one()

    return {
        "videos": videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total,
    }


@router.get("/all_from_section/{section_id: int}", response_model=VideosListResponse)
@db_transaction
def get_all_videos(section_id: int, db: Session = Depends(get_db)):
    videos = db.scalars(select(Video)
                        .where(Video.section_id == section_id)
                        .options(joinedload(Video.channel))
                        ).all()

    total = db.execute(select(func.count(Video.id))).scalar_one()

    return {
        "videos": videos,
        "total": total,
        "page": 1,
        "limit": 0,
        "has_more": False,
    }


@router.get("/{video_id}", response_model=VideoResponse)
@db_transaction
def get_video(video_id: int, db: Session = Depends(get_db)):
    video = db.execute(select(Video)
                       .where(Video.id == video_id)
                       .options(joinedload(Video.channel))
                       ).scalar_one_or_none()

    if not video or not video.path:
        logger.warning(f"В БД нет видео/информации_о_пути_видео с id={video_id}")
        raise video_exception

    full_video_path = Path(video.path)

    if not full_video_path.exists() or not full_video_path.is_dir():
        logger.warning(f"Путь к папке видео {video.name} недействителен..")
        raise video_exception

    video_file = get_video_file(full_video_path)

    if not video_file:
        logger.warning(f"У видео {video.name} нет видео-файла..")
        raise video_exception

    db.execute(update(Video).where(Video.id == video_id).values(views=Video.views + 1))

    return video


@router.get("/recommended/{video_id}", response_model=VideosListResponse)
@db_transaction
def get_recommendations(
        video_id: int,
        page: int = 1,
        limit: int = 21,
        seed: float = 0.5,
        db: Session = Depends(get_db)
):
    video = db.execute(select(Video).where(Video.id == video_id)).scalar_one_or_none()

    if not video or not video.path:
        logger.warning(f"В БД нет видео/информации_о_пути_видео с id={video_id}")
        raise video_exception

    used_id = {video.id}

    final_videos = []

    if page == 1:
        if video.section_id is not None and video.section_index is not None:
            episodes = db.scalars(select(Video)
                                  .where(and_(
                                        Video.section_id == video.section_id,
                                        Video.section_index > video.section_index))
                                  .options(joinedload(Video.channel))
                                  .order_by(Video.section_index.asc())
                                  .limit(2)).all()

            final_videos.extend(episodes)
            used_id.update(v.id for v in episodes)

        author_videos = db.scalars(select(Video)
                                   .where(and_(
                                        Video.channel_id == video.channel_id,
                                        Video.id.notin_(used_id)))
                                   .options(joinedload(Video.channel))
                                   .limit(3)
                                   ).all()
        final_videos.extend(author_videos)
        used_id.update(v.id for v in author_videos)

        similar_videos = db.scalars(select(Video)
                                    .where(and_(
                                        Video.id.notin_(used_id),
                                        Video.tags.overlap(video.tags)))
                                    .options(joinedload(Video.channel))
                                    .limit(5)
                                    ).all()

        final_videos.extend(similar_videos)
        used_id.update(v.id for v in similar_videos)

    skip = get_offset(page, limit)

    db.execute(text("SELECT setseed(:s)"), {"s": seed})

    needed = limit - len(final_videos)
    random_videos = db.scalars(select(Video)
                               .where(Video.id.notin_(list(used_id)))
                               .options(joinedload(Video.channel))
                               .order_by(func.random())
                               .offset(skip)
                               .limit(needed)
                               ).all()

    final_videos.extend(random_videos)

    total = db.scalar(select(func.count(Video.id)))

    return {
        "videos": final_videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total
    }
