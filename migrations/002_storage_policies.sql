-- ============================================================
-- Storage bucket for InBody result images / PDFs
-- 적용 방법: Supabase Dashboard > SQL Editor 에 붙여넣고 Run
-- ============================================================

-- 1) private 버킷 생성 (이미 있으면 무시)
insert into storage.buckets (id, name, public)
values ('inbody-images', 'inbody-images', false)
on conflict (id) do nothing;

-- 2) 기존 정책 정리 (재실행 시 충돌 방지)
drop policy if exists "inbody_images_select_own" on storage.objects;
drop policy if exists "inbody_images_insert_own" on storage.objects;
drop policy if exists "inbody_images_update_own" on storage.objects;
drop policy if exists "inbody_images_delete_own" on storage.objects;

-- 3) 본인 폴더({user_id}/...) 의 파일만 접근 가능
-- 경로 첫 세그먼트가 사용자의 auth.uid() 와 일치해야 함

create policy "inbody_images_select_own"
on storage.objects for select
to authenticated
using (
  bucket_id = 'inbody-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);

create policy "inbody_images_insert_own"
on storage.objects for insert
to authenticated
with check (
  bucket_id = 'inbody-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);

create policy "inbody_images_update_own"
on storage.objects for update
to authenticated
using (
  bucket_id = 'inbody-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);

create policy "inbody_images_delete_own"
on storage.objects for delete
to authenticated
using (
  bucket_id = 'inbody-images'
  and (storage.foldername(name))[1] = auth.uid()::text
);
