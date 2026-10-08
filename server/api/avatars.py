from fastapi import APIRouter, Depends

from server.api.deps import get_tenant
from server.db.models import Tenant

router = APIRouter(prefix="/api/v1/avatars", tags=["avatars"])


@router.get("")
def list_avatars(tenant: Tenant = Depends(get_tenant)) -> list:
    return []
