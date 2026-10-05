# Healthcare Mobile (Expo)

FastAPI 백엔드를 사용하는 Expo + React Native 모바일 앱입니다.
인증은 Supabase Auth SDK로 수행하고, 발급된 JWT를 `Authorization: Bearer`
헤더로 FastAPI에 전달합니다.

## 구조

```
mobile/
├── App.tsx
├── src/
│   ├── contexts/AuthContext.tsx      # Supabase 세션 상태
│   ├── lib/
│   │   ├── supabase.ts               # Supabase 클라이언트 (AsyncStorage)
│   │   └── api.ts                    # FastAPI 호출 래퍼 (JWT Bearer)
│   ├── navigation/RootNavigator.tsx  # Auth/Main 분기, 탭+스택
│   ├── screens/
│   │   ├── LoginScreen.tsx
│   │   ├── HomeScreen.tsx            # 프로필 + 최근 InBody 요약
│   │   ├── InBodyListScreen.tsx
│   │   ├── InBodyCreateScreen.tsx
│   │   ├── InBodyDetailScreen.tsx
│   │   └── AnalysisScreen.tsx        # /ai/inbody/analyze 호출
│   └── types/models.ts
└── package.json
```

## 설치

```bash
cd mobile
npm install
```

## 환경 변수

`.env.example`을 복사해서 `.env`를 만들고 값을 채워주세요.

```env
EXPO_PUBLIC_SUPABASE_URL=https://<project>.supabase.co
EXPO_PUBLIC_SUPABASE_ANON_KEY=<anon-key>
EXPO_PUBLIC_API_BASE_URL=http://<백엔드 호스트>:8000
```

- iOS 시뮬레이터에서 로컬 FastAPI를 호출할 때는 `http://localhost:8000` OK
- Android 에뮬레이터: `http://10.0.2.2:8000`
- 실기기: 같은 네트워크의 PC IP (예: `http://192.168.0.10:8000`)
  이때 FastAPI는 `uvicorn main:app --host 0.0.0.0 --port 8000`로 실행

## 실행

```bash
npm run start       # Metro + Expo Dev Tools
npm run ios         # iOS 시뮬레이터
npm run android     # Android 에뮬레이터
```

QR로 실기기 테스트는 Expo Go 앱을 사용하세요.

## 백엔드와의 연동

| 화면 | 호출 |
| --- | --- |
| Login | `supabase.auth.signInWithPassword` / `signUp` |
| Home | `GET /users/me`, `GET /inbody?limit=1` |
| InBody List | `GET /inbody?limit=50` |
| InBody Create | `POST /inbody` |
| InBody Detail | `GET /inbody/{id}`, `DELETE /inbody/{id}` |
| Analysis | `POST /ai/inbody/analyze` |

## 백엔드 CORS

FastAPI 쪽 `.env`의 `ALLOWED_ORIGINS`는 기본 `*`이라 따로 만질 필요 없지만,
운영 시에는 앱 도메인/스킴으로 좁혀주세요.
