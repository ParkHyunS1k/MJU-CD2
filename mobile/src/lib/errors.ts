// 서버·로그인 오류를 사용자가 이해할 수 있는 한국어 안내로 바꾼다.
// 원래 오류 내용은 개발자가 볼 수 있게 콘솔에 남긴다.

export class ApiError extends Error {
  status: number; // 0 = 서버에 닿지 못함(네트워크), 그 외 HTTP 상태 코드
  detail: string; // 서버가 보낸 원래 내용

  constructor(status: number, detail: string) {
    super(friendlyMessage(status, detail));
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

const HANGUL = /[가-힣]/;

// FastAPI 오류 본문 {"detail": "..."} 에서 내용만 꺼낸다
export function parseDetail(text: string): string {
  try {
    const body = JSON.parse(text);
    const detail = body?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) return detail.map((d) => d?.msg ?? '').join(', ');
  } catch {
    // JSON이 아니면 그대로
  }
  return text;
}

export function friendlyMessage(status: number, detail: string): string {
  if (status === 0) {
    return '서버에 연결할 수 없어요. 인터넷 연결을 확인하고 다시 시도해 주세요.';
  }
  // 위험도 계산에 필요한 프로필이 비어 있을 때
  if (detail.includes('성별') || detail.includes('출생연도')) {
    return '위험도를 계산하려면 성별과 출생연도가 필요해요. 설정 → 프로필 관리에서 입력해 주세요.';
  }
  // 서버가 이미 한국어로 안내한 경우(예: 인바디 결과지가 아님)는 그대로 보여 준다
  if (HANGUL.test(detail) && status < 500 && detail.length <= 80) return detail;

  if (detail === 'No fields to update') {
    return '바꾼 값을 저장하지 못했어요. 서버가 아직 이 항목(신장·근육조절 등)을 지원하지 않을 수 있어요.';
  }
  if (detail === 'No InBody records yet') return '아직 인바디 기록이 없어요. 먼저 결과지를 올려 주세요.';
  if (detail.startsWith('Unsupported file type')) {
    return '지원하지 않는 파일 형식이에요. JPG, PNG, HEIC 사진이나 PDF를 올려 주세요.';
  }
  if (detail.startsWith('File too large')) return '파일이 너무 커요. 15MB 이하로 올려 주세요.';
  if (detail === 'Empty file') return '파일이 비어 있어요. 다른 파일을 골라 주세요.';

  switch (status) {
    case 401:
    case 403:
      return '로그인이 만료됐어요. 다시 로그인해 주세요.';
    case 404:
      return '기록을 찾을 수 없어요. 이미 삭제됐을 수 있어요.';
    case 413:
      return '파일이 너무 커요. 15MB 이하로 올려 주세요.';
    case 415:
      return '지원하지 않는 파일 형식이에요. JPG, PNG, HEIC 사진이나 PDF를 올려 주세요.';
    case 422:
      return '입력한 값을 다시 확인해 주세요.';
    case 429:
      return '요청이 많아 잠시 쉬고 있어요. 1~2분 뒤에 다시 시도해 주세요.';
    case 502:
    case 503:
    case 504:
      return 'AI 분석이 잠시 원활하지 않아요. 잠시 후 다시 시도해 주세요.';
  }
  if (status >= 500) return '서버에서 문제가 생겼어요. 잠시 후 다시 시도해 주세요.';
  return '요청을 처리하지 못했어요. 다시 시도해 주세요.';
}

// Supabase 로그인·회원가입 오류
export function friendlyAuthError(e: unknown, fallback: string): string {
  const raw = e instanceof Error ? e.message : '';
  const msg = raw.toLowerCase();
  if (msg.includes('invalid login credentials')) return '이메일 또는 비밀번호가 맞지 않아요.';
  if (msg.includes('email not confirmed')) return '이메일 인증이 아직 안 됐어요. 메일함에서 인증 링크를 눌러 주세요.';
  if (msg.includes('already registered')) return '이미 가입된 이메일이에요. 로그인해 주세요.';
  if (msg.includes('invalid format') || msg.includes('invalid email')) return '이메일 형식을 확인해 주세요.';
  if (msg.includes('password should be')) return '비밀번호 조건(8자 이상, 영문·숫자 포함)을 확인해 주세요.';
  if (msg.includes('rate limit')) return '요청이 너무 많아요. 잠시 후 다시 시도해 주세요.';
  if (msg.includes('network') || msg.includes('fetch')) {
    return '서버에 연결할 수 없어요. 인터넷 연결을 확인해 주세요.';
  }
  return HANGUL.test(raw) ? raw : fallback;
}
