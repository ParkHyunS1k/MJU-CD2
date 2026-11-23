-- 기록별 질환 위험도와 계산에 쓴 입력. 읽을 때 입력이 지금 프로필·기록 값과 다르면 다시 계산한다.
-- 설계: docs/superpowers/specs/2026-11-15-risk-change-design.md
alter table public.inbody_logs
  add column disease_risks jsonb;  -- {inputs, age, diseases: [{disease, disease_key, probability, risk_level, contributions}]}
