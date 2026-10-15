-- RAG 근거 저장소. 설계: docs/superpowers/specs/2026-10-15-rag-runtime-design.md 4절
create extension if not exists vector;

create table public.rag_documents (
  id text primary key,
  title text not null,
  publisher text,
  published_year int,
  url text,
  license_note text
);

create table public.rag_chunks (
  id bigint generated always as identity primary key,
  document_id text not null references public.rag_documents on delete cascade,
  content text not null,
  heading_path text,
  page_start int,
  page_end int,
  conditions text[] not null,
  category text not null,
  content_hash text unique not null,
  embedding_model text not null,
  embedding halfvec(768) not null
);
create index rag_chunks_embedding_idx on public.rag_chunks using hnsw (embedding halfvec_cosine_ops);

alter table public.rag_documents enable row level security;
alter table public.rag_chunks enable row level security;

create function public.match_rag_chunks(query_embedding halfvec(768), match_count int, min_similarity float)
returns table (chunk_id bigint, document_id text, title text, publisher text, published_year int, url text,
               heading_path text, page_start int, page_end int, content text, similarity float)
language sql stable
set search_path = public
as $$
  select c.id, c.document_id, d.title, d.publisher, d.published_year, d.url,
         c.heading_path, c.page_start, c.page_end, c.content,
         1 - (c.embedding <=> query_embedding) as similarity
  from public.rag_chunks c join public.rag_documents d on d.id = c.document_id
  where 1 - (c.embedding <=> query_embedding) >= min_similarity
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

revoke execute on function public.match_rag_chunks(halfvec, int, float) from public, anon, authenticated;
grant execute on function public.match_rag_chunks(halfvec, int, float) to service_role;
