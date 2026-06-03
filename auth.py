from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta, timezone
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import get_db
from database_models import User
from httpExceptions import jwt_exception, db_exception

from logger import get_logger
logger = get_logger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2b")
SECRET_KEY = "v1e2f3o4x10rand10"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")


# создаем jwt-токен
def create_jwt_token(user_id):
    access_token_expires = timedelta(minutes=60 * 24 * 7)
    to_encode = {"sub": str(user_id), "exp": datetime.now(timezone.utc) + access_token_expires}
    encode_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm='HS256')

    return {"access_token": encode_jwt, "token_type": "bearer"}


# получаем пользователя по jwt-токену
def get_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise jwt_exception
    except JWTError:
        raise jwt_exception

    try:
        user = db.execute(
            select(User).where(User.id == int(user_id))
        ).scalar_one_or_none()
        if user is None:
            raise jwt_exception
        return user
    except Exception as e:
        logger.error("Ошибка БД: ", e)
        raise db_exception
