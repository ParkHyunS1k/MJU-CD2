# Healthcare API (FastAPI)

인바디 기록·질병 위험 예측·AI 분석을 제공하는 FastAPI 백엔드입니다.
DB·인증·파일 저장은 Supabase, 서버 실행은 Railway를 사용합니다.

## 배포 서버

- 주소: https://care-production-abf2.up.railway.app
- API 문서: https://care-production-abf2.up.railway.app/docs
- 배포 서버의 환경변수는 Railway 대시보드 → Variables에서 관리합니다.

## 환경 변수

`.env.example`을 복사해서 `.env`를 만들고 값을 채워주세요. `.env`는 커밋하지 않습니다.

| 변수 | 비밀 | 어디서 얻나 |
| --- | --- | --- |
| `SUPABASE_URL` | 공개 | Supabase 대시보드 → Project Settings → API |
| `SUPABASE_ANON_KEY` | 공개 | 〃 |
| `SUPABASE_SERVICE_ROLE_KEY` | **위험** (RLS 우회) | 〃 — 채팅으로 주고받지 말 것 |
| `GEMINI_API_KEY` | 비밀 | [Google AI Studio](https://aistudio.google.com/apikey)에서 각자 발급 |
| `ALLOWED_ORIGINS` | 공개 | 기본값 `*` |

Supabase 키를 보려면 Supabase 조직에 초대받아야 합니다. 팀장에게 요청하세요.

## 로컬 실행

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

http://localhost:8000/docs 에서 API를 확인할 수 있습니다.
