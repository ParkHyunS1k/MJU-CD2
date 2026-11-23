import { ApiError, parseDetail } from './errors';
import { supabase } from './supabase';
import type {
  AnalysisResponse,
  InBodyCreateInput,
  InBodyRecord,
  InBodyUpdateInput,
  InBodyUploadResponse,
  RiskTrendPoint,
  UserProfile,
  UserUpdate,
} from '../types/models';

const baseUrl = process.env.EXPO_PUBLIC_API_BASE_URL ?? 'http://10.0.2.2:8001';

// 서버가 응답하지 않을 때 화면이 "삭제 중..." 같은 상태로 영원히 멈추지 않게 요청마다 시간 제한을 둔다
const DEFAULT_TIMEOUT_MS = 30_000;
const AI_TIMEOUT_MS = 180_000; // 결과지 읽기·AI 분석은 1분 넘게 걸릴 수 있다

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (e) {
    if (controller.signal.aborted) {
      throw new Error('서버 응답이 너무 늦어요. 잠시 후 다시 시도해 주세요.');
    }
    // 서버에 닿지 못함 (Network request failed 등)
    console.warn('[api] network error', url, e);
    throw new ApiError(0, e instanceof Error ? e.message : String(e));
  } finally {
    clearTimeout(timer);
  }
}

// 오류 응답을 사용자용 안내가 담긴 ApiError로 바꾼다. 원래 내용은 콘솔에 남긴다.
async function toApiError(res: Response, path: string): Promise<ApiError> {
  const detail = parseDetail(await res.text());
  console.warn(`[api] ${res.status} ${path}: ${detail}`);
  return new ApiError(res.status, detail);
}

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new ApiError(401, 'Not authenticated');
  return { Authorization: `Bearer ${token}` };
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs: number = DEFAULT_TIMEOUT_MS,
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(await authHeader()),
    ...((init.headers as Record<string, string>) ?? {}),
  };
  const res = await fetchWithTimeout(`${baseUrl}${path}`, { ...init, headers }, timeoutMs);
  if (!res.ok) throw await toApiError(res, path);
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export interface UploadFile {
  uri: string;
  name: string;
  type: string;
}

async function uploadInBody(file: UploadFile): Promise<InBodyUploadResponse> {
  const auth = await authHeader();
  const form = new FormData();
  form.append('file', {
    uri: file.uri,
    name: file.name,
    type: file.type,
    // React Native FormData 의 file 객체 형태
  } as unknown as Blob);

  const res = await fetchWithTimeout(
    `${baseUrl}/inbody/upload`,
    { method: 'POST', headers: auth, body: form },
    AI_TIMEOUT_MS,
  );

  if (!res.ok) throw await toApiError(res, '/inbody/upload');
  return (await res.json()) as InBodyUploadResponse;
}

export const api = {
  getMe: () => request<UserProfile>('/users/me'),
  updateMe: (payload: UserUpdate) =>
    request<UserProfile>('/users/me', {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),

  listInBody: (limit = 20) =>
    request<InBodyRecord[]>(`/inbody?limit=${limit}`),
  getInBody: (id: string) => request<InBodyRecord>(`/inbody/${id}`),
  getRiskTrend: () => request<RiskTrendPoint[]>('/inbody/risk-trend'),
  createInBody: (payload: InBodyCreateInput) =>
    request<InBodyRecord>('/inbody', {
      method: 'POST',
      body: JSON.stringify(payload),
    }),
  updateInBody: (id: string, payload: InBodyUpdateInput) =>
    request<InBodyRecord>(`/inbody/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(payload),
    }),
  deleteInBody: (id: string) =>
    request<void>(`/inbody/${id}`, { method: 'DELETE' }),
  uploadInBody,

  analyzeInBody: (id?: string) =>
    request<AnalysisResponse>(
      id ? `/ai/inbody/${id}/analyze` : '/ai/inbody/analyze',
      { method: 'POST' },
      AI_TIMEOUT_MS,
    ),
};
