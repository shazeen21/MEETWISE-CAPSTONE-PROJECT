"""Role-Based Access Control (RBAC) and Security Dependencies for MeetWise AI.

Defines corporate roles:
- Admin: Full system access, keyword config, audit logs, employee management.
- Manager: Access to all meeting summaries, manager briefs, analytics, and action item tracking.
- Employee: Access to personal employee reports, meeting transcripts, and search.
"""

from enum import Enum
from typing import List, Optional
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel


class UserRole(str, Enum):
    ADMIN = "admin"
    MANAGER = "manager"
    EMPLOYEE = "employee"


class CurrentUser(BaseModel):
    user_id: str = "emp-admin"
    username: str = "Admin User"
    email: str = "admin@company.internal"
    role: UserRole = UserRole.ADMIN
    department: str = "Executive"


def get_current_user(
    x_user_id: Optional[str] = Header("emp-admin", alias="X-User-Id"),
    x_user_role: Optional[str] = Header("admin", alias="X-User-Role"),
    x_user_email: Optional[str] = Header("admin@company.internal", alias="X-User-Email"),
) -> CurrentUser:
    """Extract authenticated corporate user from headers (or defaults to Admin for local dev)."""
    try:
        role = UserRole(x_user_role.lower())
    except ValueError:
        role = UserRole.EMPLOYEE

    return CurrentUser(
        user_id=x_user_id or "emp-user",
        username=x_user_email.split("@")[0].title() if x_user_email else "Corporate User",
        email=x_user_email or "user@company.internal",
        role=role,
        department="General",
    )


def require_role(allowed_roles: List[UserRole]):
    """FastAPI dependency to enforce role permissions on sensitive routes."""
    def role_checker(current_user: CurrentUser = Header(None)) -> CurrentUser:
        # We allow fallback to get_current_user
        if current_user is None:
            current_user = CurrentUser()

        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Route requires one of: {[r.value for r in allowed_roles]}; current role is '{current_user.role.value}'.",
            )
        return current_user

    return role_checker


def require_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Convenience dependency for Admin-only routes."""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrative privileges required.",
        )
    return current_user


def require_manager_or_admin(current_user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Convenience dependency for Manager or Admin routes."""
    if current_user.role not in [UserRole.ADMIN, UserRole.MANAGER]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Managerial or Administrative privileges required.",
        )
    return current_user
