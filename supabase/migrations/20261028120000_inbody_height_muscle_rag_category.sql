-- 인바디 결과지(InBody270)의 나머지 수치 컬럼, 검색 RPC가 청크 자신의 분야를 돌려준다.
-- 설계: docs/superpowers/specs/2026-10-28-report-personalization-design.md 5.1절
-- 표준 범위·운동소비열량은 키·성별·체중으로 계산되는 참고값이라 저장하지 않는다. 회원번호(전화번호)도 저장하지 않는다.
alter table public.inbody_logs
  add column height numeric,                -- 신장(cm)
  add column muscle_control numeric,        -- 근육조절(kg). 양수면 그만큼 늘려야 표준
  add column inbody_score numeric,          -- 인바디점수
  add column target_weight numeric,         -- 적정체중(kg)
  add column weight_control numeric,        -- 체중조절(kg)
  add column fat_control numeric,           -- 지방조절(kg)
  add column fat_free_mass numeric,         -- 제지방량(kg)
  add column obesity_degree numeric,        -- 비만도(%)
  add column recommended_calories numeric,  -- 권장섭취열량(kcal)
  add column bmi_evaluation text,           -- 비만평가 BMI
  add column body_fat_evaluation text,      -- 비만평가 체지방률
  add column segmental_lean jsonb,          -- 부위별 근육분석 {left_arm, right_arm, trunk, left_leg, right_leg}
  add column segmental_fat jsonb,           -- 부위별 체지방분석(같은 키)
  add column impedance jsonb;               -- {"20kHz"|"100kHz": {RA, LA, TR, RL, LL}}

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
