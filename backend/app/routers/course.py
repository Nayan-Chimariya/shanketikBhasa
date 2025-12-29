from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, UserCourseProgress
from ..schemas import CourseProgressResponse, IncrementCourseResponse
from ..dependencies import get_current_active_user

router = APIRouter(prefix="/api", tags=["Course Progress"])


@router.post("/course/{course_name}", response_model=IncrementCourseResponse)
def increment_course_count(
    course_name: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Increment the count for a specific course/character"""
    # Get user's course progress
    progress = db.query(UserCourseProgress).filter(
        UserCourseProgress.user_id == current_user.id
    ).first()

    if not progress:
        # Create progress if it doesn't exist
        progress = UserCourseProgress(user_id=current_user.id)
        db.add(progress)
        db.commit()
        db.refresh(progress)

    # Check if course_name is a valid field
    if not hasattr(progress, course_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid course name: {course_name}"
        )

    # Increment the count
    current_count = getattr(progress, course_name)
    setattr(progress, course_name, current_count + 1)

    db.commit()
    db.refresh(progress)

    new_count = getattr(progress, course_name)

    return {
        "message": f"Successfully incremented {course_name}",
        "new_count": new_count
    }


@router.get("/course-counts", response_model=CourseProgressResponse)
def get_course_counts(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all course progress counts for the current user"""
    # Get user's course progress
    progress = db.query(UserCourseProgress).filter(
        UserCourseProgress.user_id == current_user.id
    ).first()

    if not progress:
        # Create default progress if it doesn't exist
        progress = UserCourseProgress(user_id=current_user.id)
        db.add(progress)
        db.commit()
        db.refresh(progress)

    return progress
