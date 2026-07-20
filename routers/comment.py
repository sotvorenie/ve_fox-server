from fastapi import APIRouter, Depends, Body
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import (select, func, and_,
                        delete, update, exists,
                        case)
from typing import Optional

from models import (CommentsListResponse, LikeResponse, CommentForListResponse,
                    DeletedCommentsCountResponse)
from database_models import User, Comment, CommentLike
from auth import get_user, get_safely_user
from database import get_db
from utils import db_transaction, get_offset
from httpExceptions import no_comment_exception, not_my_comment_exception

from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/comment", tags=["Comment"])


# рекурсивно подсчитываем общее количество ответов на комментарий
def get_total_replies_subquery():
    recursive_replies = (
        select(Comment.id.label("id"), Comment.parent_id.label("root_parent_id"))
        .where(Comment.parent_id.is_not(None))
        .cte(name="recursive_replies", recursive=True)
    )
    rr_alias = recursive_replies.alias()
    child_comments = (
        select(Comment.id, rr_alias.c.root_parent_id)
        .join(rr_alias, Comment.parent_id == rr_alias.c.id)
    )
    recursive_replies = recursive_replies.union_all(child_comments)
    return (
        select(
            recursive_replies.c.root_parent_id.label("parent_id"),
            func.count(recursive_replies.c.id).label("count")
        )
        .group_by(recursive_replies.c.root_parent_id)
        .subquery()
    )


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

    total_replies_query = get_total_replies_subquery()

    query = (
        select(Comment, func.coalesce(total_replies_query.c.count, 0).label("replies_count"), is_liked_expr)
        .outerjoin(total_replies_query, Comment.id == total_replies_query.c.parent_id)
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

    total_replies_query = get_total_replies_subquery()

    result = db.execute(
        select(Comment, func.coalesce(total_replies_query.c.count, 0).label("replies_count"), is_liked_expr)
        .outerjoin(total_replies_query, Comment.id == total_replies_query.c.parent_id)
        .where(Comment.parent_id == comment_id)
        .options(joinedload(Comment.user))
        .order_by(Comment.date.desc())
        .offset(skip)
        .limit(limit)
    ).unique().all()

    total_replies = db.execute(
        select(func.count(Comment.id))
        .where(Comment.parent_id == comment_id)
    ).scalar_one()

    comments = []
    for comment, count, is_liked in result:
        comment.question_comments_count = count
        comment.is_liked = is_liked
        comments.append(comment)

    return {
        "comments": comments,
        "total": total_replies,
        "page": page,
        "limit": limit,
        "has_more": (skip + limit) < total_replies,
    }


@router.post("/add/{video_id}", response_model=CommentForListResponse)
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

    new_comment.question_comments_count = 0
    new_comment.is_liked = False

    return new_comment


@router.patch("/redact/{comment_id}", response_model=CommentForListResponse)
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

    comment.question_comments_count = 0
    comment.is_liked = False

    return comment


@router.post("/delete/{comment_id}", response_model=DeletedCommentsCountResponse)
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

    recursive_comments = (
        select(Comment.id)
        .where(Comment.id == comment_id)
        .cte(name="recursive_comments", recursive=True)
    )
    alias_comments = select(recursive_comments.c.id).alias("ac")
    child_comments = select(Comment.id).join(
        alias_comments, Comment.parent_id == alias_comments.c.id
    )
    recursive_comments = recursive_comments.union_all(child_comments)
    deleted_count = db.scalar(select(func.count(recursive_comments.c.id)))

    db.delete(comment)

    return {
        "deleted_count": deleted_count
    }


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
