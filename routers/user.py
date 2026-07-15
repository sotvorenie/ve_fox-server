from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy.orm import Session
from pathlib import Path
import os
import time
import shutil

from models import (SuccessResponse, AvatarResponse, PasswordResponse,
                    RouterMapRequest, RouterMapResponse)
from utils import db_transaction
from database_models import User
from httpExceptions import empty_user_data_exception, duplication_password_exception
from auth import get_user, pwd_context
from config import AVATARS_DIRECTORY
from utils import get_file_path, get_file_url
from database import get_db

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/user", tags=["User"])


@router.post("/redact_name", response_model=SuccessResponse)
@db_transaction
def redact_user_name(name: str, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    if not name:
        raise empty_user_data_exception

    if current_user.name != name:
        current_user.name = name

    return {
        "success": True,
    }


@router.post("/check_password", response_model=SuccessResponse)
@db_transaction
def check_user_password(data: PasswordResponse, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    if not data.password:
        raise empty_user_data_exception

    check = pwd_context.verify(data.password, current_user.password)

    return {
        "success": check,
    }


@router.post("/redact_password", response_model=SuccessResponse)
@db_transaction
def redact_user_password(data: PasswordResponse, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    if not data.password:
        raise empty_user_data_exception

    check = pwd_context.verify(data.password, current_user.password)
    if check:
        raise duplication_password_exception

    current_user.password = pwd_context.hash(data.password)

    return {
        "success": True,
    }


@router.post("/redact_avatar", response_model=AvatarResponse)
@db_transaction
def redact_user_avatar(file: UploadFile, current_user: User = Depends(get_user), db: Session = Depends(get_db)):
    suffix = Path(file.filename).suffix.lower()
    file_name = f"{current_user.id}_{int(time.time())}{suffix}"
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


@router.post("/set_router_map", response_model=SuccessResponse)
@db_transaction
def set_user_router_map(
        data: RouterMapRequest,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    current_user.router_map = data.router_map

    return {"success": True}


@router.get("/get_router_map", response_model=RouterMapResponse)
@db_transaction
def get_user_router_map(
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    router_map = current_user.router_map if current_user.router_map else ''
    return {"router_map": router_map}
