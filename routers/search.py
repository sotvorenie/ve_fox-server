from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func, select, and_

from models import SearchResponse
from database_models import Channel, Video
from utils import get_offset, db_transaction, get_tags_by_title
from database import get_db

router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/", response_model=SearchResponse)
@db_transaction
def search(value: str, page: int = 1, limit: int = 21, db: Session = Depends(get_db)):
    skip = get_offset(page, limit)

    words = [w.strip() for w in value.split() if w.strip()]

    video_filters = []
    channel_names_filters = []

    for word in words:
        value_lower = f"%{word.lower()}%"

        video_filters.append(Video.name.ilike(value_lower))
        video_filters.append(func.array_to_string(Video.tags, ',').ilike(value_lower))
        channel_names_filters.append(Channel.name.ilike(value_lower))

    channels = db.scalars(
        select(Channel)
        .where(or_(*channel_names_filters))
        .limit(3)
    ).all()
    channels_ids = [c.id for c in channels]

    if channels_ids:
        video_filters.append(Video.channel_id.in_(channels_ids))

    query = (
        select(Video)
        .join(Channel, Video.channel_id == Channel.id)
        .where(or_(*video_filters))
        .options(joinedload(Video.channel))
        .distinct()
        .offset(skip)
        .limit(limit)
    )
    videos = db.scalars(query).all()

    total = db.scalar(
        select(func.count(Video.id.distinct()))
        .join(Channel, Video.channel_id == Channel.id)
        .where(or_(*video_filters))
    )

    return {
        "channels": channels,
        "videos": videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total,
    }
