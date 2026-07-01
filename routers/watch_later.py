from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, delete
from datetime import datetime, timezone

from models import SuccessResponse, IsWatchLaterResponse, VideosListResponse
from database_models import User, Video, WatchLater
from auth import get_user
from database import get_db
from utils import check_watch_later, db_transaction, get_total_and_videos_from_db
from httpExceptions import video_exception

from logger import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/watch_later", tags=["WatchLater"])


@router.post("/{video_id}", response_model=SuccessResponse)
@db_transaction
def set_watch_later(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    video_db = db.get(Video, video_id)

    if not video_db:
        logger.warning(f"Видео с id={video_id} нет в БД..")
        raise video_exception

    check_watch_later_db = check_watch_later(video_id, current_user.id, db)

    if check_watch_later_db:
        watch_later_db = db.execute(select(WatchLater)
                                    .where(and_(
                                        WatchLater.video_id == video_id,
                                        WatchLater.user_id == current_user.id))
                                    ).scalar_one_or_none()
        if watch_later_db:
            watch_later_db.date = datetime.now(timezone.utc)
    else:
        watch_later_db = WatchLater(user_id=current_user.id, video_id=video_id)
        db.add(watch_later_db)

    return {"success": True}


@router.get("/is_watch_later/{video_id}", response_model=IsWatchLaterResponse)
@db_transaction
def check_is_watch_later(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    is_watch_later = check_watch_later(video_id, current_user.id, db)

    return {"is_watch_later": is_watch_later}


@router.post("/delete/{video_id}", response_model=SuccessResponse)
@db_transaction
def delete_from_watch_later(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    video_db = db.get(Video, video_id)

    if not video_db:
        logger.warning(f"Видео с id={video_id} нет в БД..")
        raise video_exception

    db.execute(delete(WatchLater)
                        .where(and_(
                            WatchLater.video_id == video_id,
                            WatchLater.user_id == current_user.id))
                        )

    return {"success": True}


@router.get("/all", response_model=VideosListResponse)
@db_transaction
def get_list_watch_later(page: int = 1,
                         limit: int = 21,
                         current_user: User = Depends(get_user),
                         db: Session = Depends(get_db)
                         ):
    return get_total_and_videos_from_db(WatchLater, current_user.id, page, limit, db)
