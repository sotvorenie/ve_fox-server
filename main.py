from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session

from cache import start_db
from database import get_db, SessionLocal, Base, engine
from database_models import Channel, Video
from routers import (auth, channel, history,
                     like, save_time, search,
                     video, watch_later, user,
                     upload, comment)
from logger import setup_logger, get_logger
from httpExceptions import db_exception


setup_logger()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Запуск сервера..")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        start_db()
    finally:
        db.close()

    yield
    logger.info("Сервер останавливается..")
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,  # type: ignore
    allow_origins=["*"],
    allow_methods=['GET', 'POST', 'PATCH'],
    allow_headers=['*'],
)

app.include_router(auth.router)
app.include_router(channel.router)
app.include_router(history.router)
app.include_router(like.router)
app.include_router(save_time.router)
app.include_router(search.router)
app.include_router(video.router)
app.include_router(watch_later.router)
app.include_router(user.router)
app.include_router(upload.router)
app.include_router(comment.router)


# if VIDEO_DIRECTORY.exists() and VIDEO_DIRECTORY.is_dir():
#     app.mount("/static", StaticFiles(directory=VIDEO_DIRECTORY), name="static")


@app.get("/test")
def test_connection(db: Session = Depends(get_db)):
    try:
        channels_count = db.query(Channel).count()
        videos_count = db.query(Video).count()

        return {
            "message": "Сервер жив!! (пока что.. :) )",
            "channels_in_cache": channels_count,
            "videos_in_cache": videos_count,
        }
    except Exception as e:
        logger.error(f"Ошибка БД: ", e)
        raise db_exception
