from functools import lru_cache

import httpx
from supabase import Client, create_client
from supabase.lib.client_options import SyncClientOptions

from app.core.config import get_settings


def _options() -> SyncClientOptions:
    # 클라이언트 하나를 여러 스레드가 같이 쓴다(검색 RPC 병렬 호출, FastAPI 스레드풀). postgrest 기본값인
    # HTTP/2 연결을 동기 httpx로 스레드끼리 나눠 쓰면 간헐적으로 [Errno 35] Resource temporarily unavailable이 난다
    return SyncClientOptions(httpx_client=httpx.Client(http2=False, timeout=httpx.Timeout(60.0), follow_redirects=True))


@lru_cache
def get_supabase() -> Client:
    """일반 요청용 (RLS 적용). anon key 사용."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY, options=_options())


@lru_cache
def get_supabase_admin() -> Client:
    """관리자용 (RLS 우회). service_role key 사용. 서버 내부에서만."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY, options=_options())
