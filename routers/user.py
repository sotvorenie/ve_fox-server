from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.orm import Session
from pathlib import Path
import os
import time
import shutil

from models import SuccessResponse, AvatarResponse, RedactUserData
from utils import db_transaction
from database_models import User
from httpExceptions import empty_user_data_exception
from auth import get_user
from config import AVATARS_DIRECTORY
from utils import get_file_path, get_file_url
from database import get_db

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/user", tags=["User"])


@router.post("/redact_data", response_model=SuccessResponse)
@db_transaction
def redact_user_data(user_data: RedactUserData, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    if not user_data.name or not user_data.login:
        raise empty_user_data_exception

    if current_user.name != user_data.name:
        current_user.name = user_data.name
    if current_user.login != user_data.login:
        current_user.login = user_data.login

    return {
        "success": True,
    }


@router.post("/redact_avatar", response_model=AvatarResponse)
@db_transaction
def redact_user_data(file: UploadFile, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix.lower()
    file_name = f"{current_user.login}_{int(time.time())}{suffix}"
    file_path = AVATARS_DIRECTORY / file_name

    try:
        with file_path.open(mode='wb') as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    if current_user.avatar_url:
        old_avatar_file = get_file_path(current_user.avatar_url)

        if old_avatar_file and old_avatar_file.exists():
            try:
                os.remove(old_avatar_file)
            except Exception as e:
                logger.error(f"Ошибка удаления файла {old_avatar_file}: {e}")

    new_avatar_url = get_file_url(file_path)
    current_user.avatar_url = new_avatar_url

    return {
        "new_avatar_url": new_avatar_url
    }
