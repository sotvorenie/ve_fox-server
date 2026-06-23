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
    status_code=409,
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

empty_user_data_exception = HTTPException(
    status_code=400,
    detail="Неверные данные пользователя",
)

duplication_password_exception = HTTPException(
    status_code=400,
    detail="Новый пароль должен отличаться от текущего",
)

section_exception = HTTPException(
    status_code=400,
    detail="Данный плейлист не принадлежит каналу",
)

uploaded_video_exception = HTTPException(
    status_code=400,
    detail="Данные видео некорректны",
)

video_format_exception = HTTPException(
    status_code=400,
    detail="Неподдерживаемый формат видео"
)

photo_format_exception = HTTPException(
    status_code=400,
    detail="Неподдерживаемый формат фото"
)

duplication_section_exception = HTTPException(
    status_code=400,
    detail="Плейлист с таким названием уже существует"
)
