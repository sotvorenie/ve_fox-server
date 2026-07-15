from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional


class ORMModel(BaseModel):
    class Config:
        from_attributes = True


class BasePagination(BaseModel):
    total: int
    page: int
    limit: int
    has_more: bool


class UserRegister(BaseModel):
    login: str
    password: str
    name: str


class UserLogin(BaseModel):
    login: str
    password: str


class UserResponse(ORMModel):
    id: int
    name: str
    avatar_url: Optional[str] = None
    router_map: Optional[str] = None


class MeResponse(ORMModel):
    user: UserResponse
    token: str


class Token(BaseModel):
    access_token: str
    token_type: str


class ChannelResponse(ORMModel):
    id: int
    name: str
    date: datetime
    avatar_url: Optional[str] = None


class ChannelForListResponse(ORMModel):
    id: int
    name: str
    avatar_url: Optional[str] = None


class ChannelsListResponse(ORMModel):
    channels: list[ChannelForListResponse]
    total: int


class SectionResponse(ORMModel):
    id: int
    name: str
    channel_id: int


class SectionListResponse(ORMModel):
    sections: list[SectionResponse]
    total: int


class VideoForListResponse(ORMModel):
    id: int
    name: str
    date: datetime
    duration: float
    views: int
    video_url: str
    channel: ChannelForListResponse
    preview_url: Optional[str] = None
    saved_time: Optional[float] = None


class VideoResponse(ORMModel):
    id: int
    name: str
    video_url: str
    preview_url: str
    date: datetime
    duration: float
    views: int
    likes: int
    channel: ChannelForListResponse
    subtitle_url: Optional[str] = None


class FilmForListResponse(ORMModel):
    id: int
    name: str
    preview_url: Optional[str] = None
    duration: float


class FilmResponse(ORMModel):
    id: int
    video_url: str
    duration: float


class VideosListResponse(BasePagination):
    videos: list[VideoForListResponse]


class SearchResponse(BasePagination):
    channels: list[ChannelForListResponse]
    videos: list[VideoForListResponse]


class LikeResponse(ORMModel):
    is_liked: bool


class SavedTimeResponse(ORMModel):
    time: float


class SuccessResponse(ORMModel):
    success: bool


class IsWatchLaterResponse(ORMModel):
    is_watch_later: bool


class FilmsListResponse(BasePagination):
    films: list[FilmForListResponse]


class AvatarResponse(ORMModel):
    new_avatar_url: str


class PasswordResponse(ORMModel):
    password: str


class CreateSectionRequest(BaseModel):
    section_name: str


class RouterMapRequest(BaseModel):
    router_map: Optional[str] = None


class RouterMapResponse(ORMModel):
    router_map: Optional[str] = None
