from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select, and_, delete

from models import SuccessResponse, SavedTimeResponse
from database_models import User, SavedTime, Video
from auth import get_user
from database import get_db
from utils import db_transaction
from httpExceptions import video_exception

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/save_time", tags=["SaveTime"])


@router.post("/set/{video_id}", response_model=SuccessResponse)
@db_transaction
def save_time(video_id: int,
              time: float,
              current_user: User = Depends(get_user),
              db: Session = Depends(get_db)):
    saved_entry = db.execute(select(SavedTime)
                             .where(and_(SavedTime.user_id == current_user.id, SavedTime.video_id == video_id))
                             ).scalar_one_or_none()

    if saved_entry:
        saved_entry.time = time
    else:
        new_save = SavedTime(user_id=current_user.id, video_id=video_id, time=time)
        db.add(new_save)

    return {'success': True}


@router.post("/delete/{video_id}", response_model=SuccessResponse)
@db_transaction
def delete_time(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    db.execute(delete(SavedTime).where(and_(SavedTime.video_id == video_id, SavedTime.user_id == current_user.id)))

    return {'success': True}


@router.get("/get/{video_id}", response_model=SavedTimeResponse)
@db_transaction
def get_time(video_id: int, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    video_db = db.get(Video, video_id)
    if not video_db:
        logger.warning(f"Видео с id={video_id} нет в БД..")
        raise video_exception

    time = 0

    saved_query = db.execute(select(SavedTime)
                             .where(and_(SavedTime.user_id == current_user.id, SavedTime.video_id == video_id))
                             ).scalar_one_or_none()

    if saved_query and saved_query.time:
        time = saved_query.time

    return {'time': time}
