from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func, exists
from pathlib import Path

from database import get_db
from models import (ChannelResponse, ChannelsListResponse, SectionResponse,
                    VideosListResponse, SuccessResponse, CreateSectionRequest,
                    SectionListResponse)
from database_models import (Channel, ChannelSection, Video,
                             User)
from utils import get_offset, db_transaction
from httpExceptions import channel_exception, duplication_section_exception
from auth import get_user

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/channel", tags=["Channel"])


@router.get("/all", response_model=ChannelsListResponse)
@db_transaction
def get_channels(db: Session = Depends(get_db)):
    channels = db.scalars(select(Channel)).all()

    total = db.execute(select(func.count(Channel.id))).scalar_one()

    return {
        "channels": channels,
        "total": total
    }


@router.get("/{channel_id}/sections", response_model=SectionListResponse)
@db_transaction
def get_channel_sections(channel_id: int, db: Session = Depends(get_db)):
    sections = db.scalars(
        select(ChannelSection)
        .where(ChannelSection.channel_id == channel_id)
    ).all()

    total = db.execute(
        select(func.count(ChannelSection.id))
        .where(ChannelSection.channel_id == channel_id)
    ).scalar_one()

    return {
        "sections": sections,
        "total": total
    }


@router.get("/{channel_id}/has_sections", response_model=SuccessResponse)
@db_transaction
def check_has_channel_sections(channel_id: int, db: Session = Depends(get_db)):
    has_sections = db.scalar(
        select(
            exists().where(ChannelSection.channel_id == channel_id)
        )
    )

    return {
        "success": has_sections,
    }


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


@router.post("/{channel_id}/create_section", response_model=SectionResponse)
@db_transaction
def create_section(
        channel_id: int,
        data: CreateSectionRequest,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    has_section_with_this_name = db.scalar(
        select(
            exists().where(
                ChannelSection.channel_id == channel_id,
                ChannelSection.name == data.section_name,
            )
        )
    )
    if has_section_with_this_name:
        raise duplication_section_exception

    new_section = ChannelSection(
        name=data.section_name,
        channel_id=channel_id,
    )
    db.add(new_section)
    db.flush()

    return new_section
