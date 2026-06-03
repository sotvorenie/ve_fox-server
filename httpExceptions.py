from fastapi import HTTPException

jwt_exception = HTTPException(
        status_code=401,
        detail="Не удалось валидировать токен",
        headers={"WWW-Authenticate": "Bearer"},
)

db_exception = HTTPException(
    status_code=500,
    detail="Ошибка БД",
)

registration_exception = HTTPException(
    status_code=400,
    detail="Пользователь с таким логином уже существует",
)

auth_exception = HTTPException(
    status_code=401,
    detail="Неверное имя или пароль",
)

channel_exception = HTTPException(
    status_code=404,
    detail="Канал не найден",
)

video_exception = HTTPException(
    status_code=404,
    detail="Видео не найдено",
)
