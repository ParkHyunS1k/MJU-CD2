from app.core import supabase as sb


def test_clients_use_http1_with_generous_timeout(monkeypatch):
    # 여러 스레드가 클라이언트 하나를 같이 쓴다(검색 RPC 병렬 호출, FastAPI 스레드풀).
    # 동기 httpx의 HTTP/2 연결을 스레드끼리 나눠 쓰면 간헐적으로 [Errno 35]가 난다
    sb.get_supabase_admin.cache_clear()
    client = sb.get_supabase_admin()
    session = client.postgrest.session
    assert session._transport._pool._http2 is False
    assert session.timeout.read >= 60
    sb.get_supabase_admin.cache_clear()
