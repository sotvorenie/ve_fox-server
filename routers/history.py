from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, func, delete
from datetime import datetime, timezone

from models import SuccessResponse, VideosListResponse
from database_models import User, Video, History
from auth import get_user
from database import get_db
from utils import db_transaction, get_total_and_videos_from_db
from httpExceptions import video_exception

from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


@router.post("/set/{video_id}", response_model=SuccessResponse)
@db_transaction
def set_to_history(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    video_db = db.get(Video, video_id)
    if not video_db:
        logger.warning(f"Видео с id={video_id} нет в БД..")
        raise video_exception

    history_entry = (db.execute(select(History)
                                .where(
                                    History.user_id == current_user.id,
                                    History.video_id == video_id
                                ))
                     .scalar_one_or_none())

    if history_entry:
        history_entry.date = datetime.now(timezone.utc)
    else:
        new_history = History(user_id=current_user.id, video_id=video_id)
        db.add(new_history)

    db.flush()

    total = db.scalar(select(func.count(History.id)).where(History.user_id == current_user.id))
    if total > 100:
        oldest_id = db.scalar(
            select(History.id)
            .where(History.user_id == current_user.id)
            .order_by(History.date.asc())
            .limit(1)
        )
        db.execute(delete(History).where(History.id == oldest_id))

    return {'success': True}


@router.get("/all", response_model=VideosListResponse)
@db_transaction
def get_list_history(page: int = 1,
                     limit: int = 21,
                     current_user: User = Depends(get_user),
                     db: Session = Depends(get_db)):
    return get_total_and_videos_from_db(History, current_user.id, page, limit, db)
