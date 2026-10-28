-- 리포트 개인화: 신장·근육조절 컬럼, 검색 RPC가 청크 자신의 분야를 돌려준다.
-- 설계: docs/superpowers/specs/2026-10-28-report-personalization-design.md 5.1절
alter table public.inbody_logs
  add column height numeric,
  add column muscle_control numeric;

-- 반환 형식이 바뀌므로 지우고 다시 만든다
drop function public.match_rag_chunks(extensions.halfvec, int, float);

create function public.match_rag_chunks(query_embedding extensions.halfvec(768), match_count int, min_similarity float)
returns table (chunk_id bigint, document_id text, title text, publisher text, published_year int, url text,
               heading_path text, page_start int, page_end int, content text, category text, similarity float)
language sql stable
set search_path = public, extensions
as $$
  select c.id, c.document_id, d.title, d.publisher, d.published_year, d.url,
         c.heading_path, c.page_start, c.page_end, c.content, c.category,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.rag_chunks c join public.rag_documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

revoke execute on function public.match_rag_chunks(extensions.halfvec, int, float) from public, anon, authenticated;
grant execute on function public.match_rag_chunks(extensions.halfvec, int, float) to service_role;
