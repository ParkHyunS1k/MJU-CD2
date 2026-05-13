from uuid import uuid4

from app.core.supabase import get_supabase_admin

BUCKET = "inbody-images"

EXT_BY_MIME = {
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/heic": "heic",
    "application/pdf": "pdf",
}


def upload_inbody_file(user_id: str, file_bytes: bytes, content_type: str) -> str:
    """업로드 후 storage path 를 반환한다. 형식: '{user_id}/{uuid}.{ext}'"""
    ext = EXT_BY_MIME.get(content_type)
    if ext is None:
        raise ValueError(f"Unsupported content type: {content_type}")

    path = f"{user_id}/{uuid4()}.{ext}"
    sb = get_supabase_admin()
    sb.storage.from_(BUCKET).upload(
        path=path,
        file=file_bytes,
        file_options={"content-type": content_type, "upsert": "false"},
    )
    return path


def delete_inbody_file(path: str) -> None:
    sb = get_supabase_admin()
    sb.storage.from_(BUCKET).remove([path])
