from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, func, select
import json
from typing import List

from models import (SearchResponse, SearchHistoryRequest, SuccessResponse,
                    SearchHistoryResponse)
from database_models import Channel, Video, User
from utils import get_offset, db_transaction
from database import get_db
from auth import get_user

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


@router.post("/set_history", response_model=SuccessResponse)
@db_transaction
def set_to_search_history(
        data: SearchHistoryRequest,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    new_query = data.search.strip()
    if not new_query:
        return {"success": False}

    if current_user.search_history:
        history: List[str] = json.loads(current_user.search_history)
    else:
        history = []

    if new_query in history:
        history.remove(new_query)

    history.insert(0, new_query)
    history = history[:10]

    current_user.search_history = json.dumps(history)

    return {"success": True}


@router.get("/get_history", response_model=SearchHistoryResponse)
@db_transaction
def get_search_history(
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    history = json.loads(current_user.search_history) if current_user.search_history else []

    return {"search_history": history}


@router.post("/delete_from_history", response_model=SuccessResponse)
@db_transaction
def delete_from_search_history(
        data: SearchHistoryRequest,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    history = json.loads(current_user.search_history) if current_user.search_history else []
    if data.search in history:
        history.remove(data.search)
    current_user.search_history = json.dumps(history)

    return {"success": True}

