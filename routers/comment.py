from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import (select, func, and_,
                        delete, update, exists,
                        case)
from typing import Optional

from models import (CommentsListResponse, SuccessResponse, CommentBaseResponse,
                    LikeResponse)
from database_models import User, Comment, CommentLike
from auth import get_user, get_safely_user
from database import get_db
from utils import db_transaction, get_offset
from httpExceptions import no_comment_exception, not_my_comment_exception

from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/comment", tags=["Comment"])


@router.get("/{video_id}", response_model=CommentsListResponse)
@db_transaction
def get_video_comments(
        video_id: int,
        is_new: bool = True,
        page: int = 1,
        limit: int = 21,
        current_user: Optional[User] = Depends(get_safely_user),
        db: Session = Depends(get_db)
):
    skip = get_offset(page, limit)

    if current_user:
        is_liked_expr = case(
            (Comment.user_id == current_user.id, False),
            else_=exists().where(
                and_(
                    CommentLike.comment_id == Comment.id,
                    CommentLike.user_id == current_user.id
                )
            )
        ).label("is_liked")
    else:
        is_liked_expr = func.lit(False).label("is_liked")

    replies_count_query = (
        select(Comment.parent_id, func.count(Comment.id).label("count"))
        .where(Comment.parent_id.is_not(None))
        .group_by(Comment.parent_id)
        .subquery()
    )

    query = (
        select(Comment, func.coalesce(replies_count_query.c.count, 0).label("replies_count"), is_liked_expr)
        .outerjoin(replies_count_query, Comment.id == replies_count_query.c.parent_id)
        .where(and_(Comment.video_id == video_id, Comment.parent_id.is_(None)))
        .options(joinedload(Comment.user))
        .order_by(Comment.date.desc() if is_new else Comment.likes.desc())
        .offset(skip)
        .limit(limit)
    )

    result = db.execute(query).unique().all()

    total = db.execute(select(func.count(Comment.id)).where(Comment.video_id == video_id)).scalar_one()

    comments = []
    for comment, count, is_liked in result:
        comment.question_comments_count = count
        comment.is_liked = is_liked
        comments.append(comment)

    return {
        "comments": comments,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total
    }


@router.get("/answers/{comment_id}", response_model=CommentsListResponse)
@db_transaction
def get_comment_answers(
        comment_id: int,
        page: int = 1,
        limit: int = 21,
        current_user: Optional[User] = Depends(get_safely_user),
        db: Session = Depends(get_db)
):
    skip = get_offset(page, limit)

    if current_user:
        is_liked_expr = case(
            (Comment.user_id == current_user.id, False),
            else_=exists().where(
                and_(
                    CommentLike.comment_id == Comment.id,
                    CommentLike.user_id == current_user.id
                )
            )
        ).label("is_liked")
    else:
        is_liked_expr = func.lit(False).label("is_liked")

    replies_count_query = (
        select(Comment.parent_id, func.count(Comment.id).label("count"))
        .where(Comment.parent_id.is_not(None))
        .group_by(Comment.parent_id)
        .subquery()
    )

    result = db.execute(
        select(Comment, func.coalesce(replies_count_query.c.count, 0).label("replies_count"), is_liked_expr)
        .outerjoin(replies_count_query, Comment.id == replies_count_query.c.parent_id)
        .where(Comment.parent_id == comment_id)
        .options(joinedload(Comment.user))
        .order_by(Comment.date.desc())
        .offset(skip)
        .limit(limit)
    ).unique().all()

    comments = []
    for comment, count, is_liked in result:
        comment.question_comments_count = count
        comment.is_liked = is_liked
        comments.append(comment)

    return {
        "comments": comments,
        "total": len(comments),
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < len(comments)
    }


@router.post("/add/{video_id}", response_model=CommentBaseResponse)
@db_transaction
def add_comment(
        video_id: int,
        text: str = Body(..., embed=True),
        parent_id: Optional[int] = Body(None, embed=True),
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    redacted_comment_text = text.strip()[:1000]
    new_comment = Comment(
        text=redacted_comment_text,
        is_redacted=False,
        likes=0,
        parent_id=parent_id,
        user_id=current_user.id,
        video_id=video_id,
    )
    db.add(new_comment)
    db.flush()
    db.refresh(new_comment)

    return new_comment


@router.patch("/redact/{comment_id}", response_model=CommentBaseResponse)
@db_transaction
def edit_comment(
        comment_id: int,
        text: str = Body(..., embed=True),
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    comment = db.get(Comment, comment_id)

    if not comment:
        raise no_comment_exception

    if comment.user_id != current_user.id:
        raise not_my_comment_exception

    redacted_comment_text = text.strip()[:1000]
    comment.text = redacted_comment_text
    comment.is_redacted = True

    db.flush()
    db.refresh(comment)

    return comment


@router.post("/delete/{comment_id}", response_model=SuccessResponse)
@db_transaction
def delete_comment(
        comment_id: int,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    comment = db.get(Comment, comment_id)
    if not comment:
        raise no_comment_exception

    if comment.user_id != current_user.id:
        raise not_my_comment_exception

    db.delete(comment)

    return {"success": True}


@router.post("/like/{comment_id}", response_model=LikeResponse)
@db_transaction
def like_comment(
        comment_id: int,
        current_user: User = Depends(get_user),
        db: Session = Depends(get_db)
):
    comment = db.get(Comment, comment_id)
    if not comment:
        raise no_comment_exception

    result = db.execute(
        delete(CommentLike)
        .where(and_(
            CommentLike.comment_id == comment_id,
            CommentLike.user_id == current_user.id
        )))

    if result.rowcount == 0:
        new_like = CommentLike(user_id=current_user.id, comment_id=comment_id)
        db.add(new_like)
        db.execute(update(Comment).where(Comment.id == comment_id).values(likes=Comment.likes + 1))
        is_liked = True
    else:
        db.execute(update(Comment).where(Comment.id == comment_id).values(likes=Comment.likes - 1))
        is_liked = False

    return {"is_liked": is_liked}
