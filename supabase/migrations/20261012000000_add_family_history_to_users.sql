-- 질병 위험 예측 입력: 부·모·형제 중 진단자 여부. 가입 시 예/아니오/모름으로 받고 모름은 null.
-- 설계: docs/superpowers/specs/2026-10-12-risk-model-redesign-design.md 3.3, 6.2
alter table public.users
  add column fh_hypertension boolean,
  add column fh_diabetes boolean,
  add column fh_dyslipidemia boolean;

comment on column public.users.fh_hypertension is '부·모·형제 중 고혈압 진단자. null=모름';
comment on column public.users.fh_diabetes is '부·모·형제 중 당뇨 진단자. null=모름';
comment on column public.users.fh_dyslipidemia is '부·모·형제 중 고지혈증(이상지질혈증) 진단자. null=모름';
