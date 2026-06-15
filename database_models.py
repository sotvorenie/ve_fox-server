from sqlalchemy import (Column, Integer, String,
                        ForeignKey, DateTime, Table,
                        UniqueConstraint, Index, text)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import func
from datetime import datetime
from typing import Optional, List

from database import Base


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(20))
    login: Mapped[str] = mapped_column(String(20), unique=True)
    password: Mapped[str] = mapped_column(String(255))
    avatar_url: Mapped[Optional[str]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(),
                                                 nullable=False,
                                                 onupdate=func.now()
                                                 )

    likes: Mapped[List["Like"]] = relationship(back_populates='user', cascade='all, delete-orphan')
    history: Mapped[List["History"]] = relationship(back_populates='user', cascade='all, delete-orphan')
    saved_times: Mapped[List["SavedTime"]] = relationship(back_populates='user', cascade='all, delete-orphan')
    watch_later: Mapped[List["WatchLater"]] = relationship(back_populates='user', cascade='all, delete-orphan')


class Channel(Base):
    __tablename__ = 'channels'

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str] = mapped_column()
    avatar_url: Mapped[Optional[str]] = mapped_column(nullable=True)
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"), nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(),
                                                 nullable=False,
                                                 onupdate=func.now()
                                                 )

    videos: Mapped[List["Video"]] = relationship(back_populates='channel', cascade='all, delete-orphan')
    sections: Mapped[List["ChannelSection"]] = relationship(back_populates='channel', cascade='all, delete-orphan')


class ChannelSection(Base):
    __tablename__ = 'channel_sections'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    channel_id: Mapped[int] = mapped_column(ForeignKey('channels.id', ondelete='CASCADE'))

    channel: Mapped[Channel] = relationship(back_populates='sections')
    videos: Mapped[List["Video"]] = relationship(back_populates='section')


class Video(Base):
    __tablename__ = 'videos'
    __table_args__ = (
        Index('idx_video_tags', 'tags', postgresql_using='gin'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(String, unique=True)
    name: Mapped[str] = mapped_column()
    video_url: Mapped[str] = mapped_column()
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration: Mapped[int] = mapped_column()
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), server_default=text("'{}'"), nullable=False)
    views: Mapped[int] = mapped_column(default=0, server_default='0', nullable=False)
    likes: Mapped[int] = mapped_column(default=0, server_default='0', nullable=False)
    preview_url: Mapped[Optional[str]] = mapped_column(nullable=True)
    subtitle_url: Mapped[Optional[str]] = mapped_column(nullable=True)
    section_index: Mapped[int] = mapped_column(default=1)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(),
                                                 nullable=False,
                                                 onupdate=func.now()
                                                 )

    channel_id: Mapped[int] = mapped_column(ForeignKey('channels.id', ondelete='CASCADE'), index=True)
    section_id: Mapped[Optional[str]] = mapped_column(ForeignKey('channel_sections.id', ondelete='SET NULL'),
                                                   nullable=True
                                                   )

    channel: Mapped["Channel"] = relationship(back_populates='videos')
    section: Mapped["ChannelSection"] = relationship(back_populates='videos')


film_actor_association = Table(
    'film_actor_association',
    Base.metadata,
    Column('film_id', Integer, ForeignKey('films.id', ondelete='CASCADE'), primary_key=True),
    Column('actor_id', Integer, ForeignKey('film_actors.id'), primary_key=True)
)


class Film(Base):
    __tablename__ = 'films'

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str] = mapped_column(unique=True)
    name: Mapped[str] = mapped_column()
    video_url: Mapped[str] = mapped_column()
    duration: Mapped[int] = mapped_column()
    preview_url: Mapped[Optional[str]] = mapped_column(nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.now(),
                                                 nullable=False,
                                                 onupdate=func.now()
                                                 )

    genre_id: Mapped[int] = mapped_column(ForeignKey('film_genres.id', ondelete='RESTRICT'))

    genre: Mapped["FilmGenre"] = relationship(back_populates='films')
    actors: Mapped[List["FilmActor"]] = relationship(secondary=film_actor_association, back_populates='films')


class FilmGenre(Base):
    __tablename__ = 'film_genres'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    films = relationship('Film', back_populates='genre')


class FilmActor(Base):
    __tablename__ = 'film_actors'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    films: Mapped[List["Film"]] = relationship(secondary=film_actor_association, back_populates='actors')


class Like(Base):
    __tablename__ = 'likes'

    __table_args__ = (UniqueConstraint('user_id', 'video_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    video_id: Mapped[int] = mapped_column(ForeignKey('videos.id', ondelete='CASCADE'), index=True)

    user: Mapped["User"] = relationship(back_populates='likes')
    video: Mapped["Video"] = relationship()


class History(Base):
    __tablename__ = 'histories'

    __table_args__ = (UniqueConstraint('user_id', 'video_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    video_id: Mapped[int] = mapped_column(ForeignKey('videos.id', ondelete='CASCADE'), index=True)

    user: Mapped["User"] = relationship(back_populates='history')
    video: Mapped["Video"] = relationship()


class SavedTime(Base):
    __tablename__ = 'saved_times'

    __table_args__ = (UniqueConstraint('user_id', 'video_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    time: Mapped[float] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    video_id: Mapped[int] = mapped_column(ForeignKey('videos.id', ondelete='CASCADE'), index=True)

    user: Mapped["User"] = relationship(back_populates='saved_times')
    video: Mapped["Video"] = relationship()


class WatchLater(Base):
    __tablename__ = 'watch_later'

    __table_args__ = (UniqueConstraint('user_id', 'video_id'),)

    id: Mapped[int] = mapped_column(primary_key=True)
    date: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'))
    video_id: Mapped[int] = mapped_column(ForeignKey('videos.id', ondelete='CASCADE'), index=True)

    user: Mapped["User"] = relationship(back_populates='watch_later')
    video: Mapped["Video"] = relationship()
