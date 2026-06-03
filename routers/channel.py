from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func
from pathlib import Path

from database import get_db
from models import ChannelResponse, SectionResponse, VideosListResponse
from database_models import Channel, ChannelSection, Video
from utils import get_offset, db_transaction
from httpExceptions import channel_exception

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/channel", tags=["Channel"])


@router.get("/all", response_model=list[ChannelResponse])
@db_transaction
def get_channels(db: Session = Depends(get_db)):
    channels = db.scalars(select(Channel)).all()
    return channels


@router.get("/{channel_id}/sections", response_model=SectionResponse)
@db_transaction
def get_channel_sections(channel_id: int, db: Session = Depends(get_db)):
    sections = db.scalars(
        select(ChannelSection)
        .where(ChannelSection.channel_id == channel_id)
    ).all()
    return sections


@router.get("/{channel_id}", response_model=ChannelResponse)
@db_transaction
def get_channel(channel_id: int, db: Session = Depends(get_db)):
    channel_db = db.get(Channel, channel_id)

    if not channel_db or not channel_db.path:
        logger.warning(f"В БД нет канала/информации_о_пути_канала с id={channel_id}")
        raise channel_exception

    if not Path(channel_db.path).exists():
        logger.warning(f"Путь к папке канала{channel_db.name} недействителен..")
        raise channel_exception

    return channel_db


@router.get("/{channel_id}/videos", response_model=VideosListResponse)
@db_transaction
def get_channel_videos(channel_id: int,
                       page: int = 1,
                       limit: int = 21,
                       is_new: bool = True,
                       db: Session = Depends(get_db)):
    skip = get_offset(page, limit)

    channel = db.get(Channel, channel_id)

    if not channel or not channel.path:
        logger.warning(f"В БД нет канала/информации_о_пути_канала с id={channel_id}")
        raise channel_exception

    if not Path(channel.path).exists():
        logger.warning(f"Путь к папке канала{channel.name} недействителен..")
        raise channel_exception

    query = select(Video).where(Video.channel_id == channel_id)

    if is_new:
        query = query.order_by(Video.date.desc())
    else:
        query = query.order_by(Video.date.asc())

    videos = db.scalars(query
                        .offset(skip)
                        .limit(limit)
                        ).all()

    total = db.scalar(
        select(func.count(Video.id))
        .where(Video.channel_id == channel_id)
    )

    return {
        "videos": videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total,
    }
