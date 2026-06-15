from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select

from models import UserRegister, MeResponse, UserLogin
from database import get_db
from auth import pwd_context, create_jwt_token, get_user
from database_models import User
from utils import db_transaction
from httpExceptions import registration_exception, auth_exception

from logger import get_logger
logger = get_logger(__name__)


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", response_model=MeResponse)
@db_transaction
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.execute(select(User).where(User.login == user_data.login)).scalar_one_or_none()
    if existing_user:
        raise registration_exception

    hashed_password = pwd_context.hash(user_data.password)

    new_user = User(name=user_data.name, password=hashed_password, login=user_data.login)
    db.add(new_user)
    db.flush()

    logger.info(f"Пользователь {user_data.name} зарегистрировался в приложении")

    new_token = create_jwt_token(new_user.id)

    return {
        "user": new_user,
        "token": new_token["access_token"],
    }


@router.post("/login", response_model=MeResponse)
@db_transaction
def auth(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.execute(select(User).where(User.login == user_data.login)).scalar_one_or_none()

    if not user or not pwd_context.verify(user_data.password, user.password):
        raise auth_exception

    logger.info(f"Пользователь {user.name} авторизовался в приложении")

    new_token = create_jwt_token(user.id)

    return {
        "user": user,
        "token": new_token["access_token"],
    }


@router.get("/me", response_model=MeResponse)
def get_me(current_user: User = Depends(get_user)):
    new_token = create_jwt_token(current_user.id)
    return {
        "user": current_user,
        "token": new_token["access_token"],
    }
