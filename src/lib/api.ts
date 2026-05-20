import { supabase } from './supabase';
import type {
  AnalysisResponse,
  InBodyCreateInput,
  InBodyRecord,
  InBodyUpdateInput,
  InBodyUploadResponse,
  UserProfile,
  UserUpdate,
} from '../types/models';

const baseUrl = process.env.EXPO_PUBLIC_API_BASE_URL;

if (!baseUrl) {
  throw new Error('Missing EXPO_PUBLIC_API_BASE_URL. Set it in mobile/.env');
}

async function authHeader(): Promise<Record<string, string>> {
  const { data } = await supabase.auth.getSession();
  const token = data.session?.access_token;
  if (!token) throw new Error('Not authenticated');
  return { Authorization: `Bearer ${token}` };
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(await authHeader()),
    ...((init.headers as Record<string, string>) ?? {}),
  };
  const res = await fetch(`${baseUrl}${path}`, { ...init, headers });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
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

  const res = await fetch(`${baseUrl}/inbody/upload`, {
    method: 'POST',
    headers: auth,
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
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

  analyzeInBody: () =>
    request<AnalysisResponse>('/ai/inbody/analyze', { method: 'POST' }),
};
