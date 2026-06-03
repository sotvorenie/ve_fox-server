from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import delete, update, and_, select, exists

from models import LikeResponse, VideosListResponse
from database_models import User, Video, Like
from auth import get_user
from database import get_db
from utils import db_transaction, get_total_and_videos_from_db
from httpExceptions import video_exception

from logger import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/like", tags=["Like"])


@router.post("/{video_id}", response_model=LikeResponse)
@db_transaction
def like(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    video_db = db.get(Video, video_id)
    if not video_db:
        logger.warning(f"Видео с id={video_id} нет в БД..")
        raise video_exception

    result = db.execute(delete(Like).where(and_(Like.user_id == current_user.id, Like.video_id == video_id)))

    if result.rowcount == 0:
        new_like = Like(user_id=current_user.id, video_id=video_id)
        db.add(new_like)
        db.execute(update(Video).where(Video.id == video_id).values(likes=Video.likes + 1))
        is_liked = True
    else:
        db.execute(update(Video).where(Video.id == video_id).values(likes=Video.likes - 1))
        is_liked = False

    return {
        "is_liked": is_liked
    }


@router.get("/is_liked/{video_id}", response_model=LikeResponse)
@db_transaction
def is_like(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    is_liked = db.scalar(select(exists().where(and_(Like.video_id == video_id, Like.user_id == current_user.id))))

    return {
        "is_liked": is_liked
    }


@router.get("/all", response_model=VideosListResponse)
@db_transaction
def get_list_likes(page: int = 1,
                   limit: int = 21,
                   current_user: User = Depends(get_user),
                   db: Session = Depends(get_db)):
    return get_total_and_videos_from_db(Like, current_user, page, limit, db)
