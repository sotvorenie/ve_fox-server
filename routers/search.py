from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func, select

from models import SearchResponse
from database_models import Channel, Video
from utils import get_offset, db_transaction, get_tags_by_title
from database import get_db


router = APIRouter(prefix="/search", tags=["Search"])


@router.get("/", response_model=SearchResponse)
@db_transaction
def search(value: str, page: int = 1, limit: int = 21, db: Session = Depends(get_db)):
    skip = get_offset(page, limit)

    value_lower = f"%{value.lower()}%"
    tags = get_tags_by_title(value)

    channels = db.scalars(select(Channel).where(Channel.name.ilike(value_lower)).limit(3)).all()
    channels_id = [c.id for c in channels]

    filters = []
    if tags:
        filters.append(Video.tags.overlap(tags))
    if channels_id:
        filters.append(Channel.id.in_(channels_id))
    if not filters:
        filters.append(Video.name.ilike(value_lower))

    videos = db.scalars(select(Video)
                        .where(or_(*filters))
                        .options(joinedload(Video.channel))
                        .offset(skip)
                        .limit(limit)
                        ).all()

    total = db.scalar(select(func.count(Video.id)).where(or_(*filters)))

    return {
        "channels": channels,
        "videos": videos,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total,
    }
