"""리포트 미리보기와 개인화 비교. 앱과 같은 흐름(개인 지표 → 특성·질환 쿼리 → 검색 → GPT 리포트 → 인용 검증)으로
KNHANES 사례(+ 선택한 인바디 기록)의 리포트를 만들고, 바꾸기 전 결과와 비교한다. 저장소 루트에서

    python -m rag.eval.report_preview run --out rag/eval/preview/after.json [--inbody-json 기록.json] [--synthetic]
    python -m rag.eval.report_preview compare rag/eval/preview/before.json rag/eval/preview/after.json

설계: docs/superpowers/specs/2026-10-28-report-personalization-design.md 5.5절. 결과물은 git 제외(개인 원자료 포함).
--inbody-json: [{"case": 이름, "profile": {gender, birth_year, fh_*}, "inbody": {...}}, ...]
--with-muscle: KNHANES에는 근육조절이 없어, 사례마다 합성 근육조절(0.0 적정, 2.0 부족 번갈아)을 넣어 체성분 유형을 만든다.
--with-previous: 사례마다 1년 전 합성 기록(좋아짐·나빠짐·비슷함 순환)을 붙여 위험도 변화(risk_change)까지 넣는다.
"""

import argparse
import json
import re
import sys
import time
from itertools import combinations
from pathlib import Path

BADGE = {"low": "낮음", "medium": "주의", "high": "높음"}
CATEGORY = {"nutrition": "식사", "exercise": "운동", "sleep": "수면", "habit": "습관"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
PREVIOUS_MEASURED_AT = "2025-11-01T00:00:00+00:00"
# 직전 기록 = 지금 값 + 이만큼. 좋아짐(지난번이 더 높음), 나빠짐, 비슷함(1.0%p 미만 예상) 순으로 돌린다
_PREVIOUS_DELTAS = [(1.5, 3.0), (-1.5, -3.0), (0.1, -0.2)]


def jaccard_mean(sets: list[set]) -> float:
    """사람들 사이 인용 청크 겹침. 쌍별 Jaccard의 평균(둘 다 비면 0으로 본다)."""
    pairs = list(combinations(sets, 2))
    if not pairs:
        return 0.0
    return sum(len(a & b) / len(a | b) if a | b else 0.0 for a, b in pairs) / len(pairs)


def _numbers(obj) -> set[str]:
    if isinstance(obj, bool):
        return set()
    if isinstance(obj, int | float):
        return {f"{float(obj):.1f}".removesuffix(".0"), f"{float(obj):.1f}"}
    if isinstance(obj, dict):
        return set().union(*map(_numbers, obj.values())) if obj else set()
    if isinstance(obj, list):
        return set().union(*map(_numbers, obj)) if obj else set()
    return set()


def mentions_user_number(text: str, user_values) -> bool:
    """행동 문장에 이 사람의 수치(인바디·개인 지표)가 그대로 들어 있는가."""
    return bool(set(_NUMBER.findall(text)) & _numbers(user_values))


# KNHANES 부위별 BIA(kg) → 결과지 부위 키. 몸통 근육은 제지방량에서 팔다리 근육을 뺀 근사값
_LEAN = {"right_arm": "BIA_LRA", "left_arm": "BIA_LLA", "right_leg": "BIA_LRL", "left_leg": "BIA_LLL"}
_FAT = {"right_arm": "BIA_FRA", "left_arm": "BIA_FLA", "trunk": "BIA_FT", "right_leg": "BIA_FRL", "left_leg": "BIA_FLL"}


def segmental_labels(df) -> list[tuple[dict, dict]]:
    """합성 데이터: 부위 kg ÷ 키² 를 성별 분포에서 하위 20%면 표준이하, 상위 20%면 표준이상으로 매긴다.
    인바디의 실제 판정 기준은 공개되지 않아 분포로 흉내 낸 것이다. 행 순서대로 (근육, 체지방)."""
    h2 = (df.HE_ht / 100) ** 2
    lean = {k: df[c] for k, c in _LEAN.items()}
    lean["trunk"] = df.BIA_FFM - sum(lean.values())
    columns = {("lean", k): v / h2 for k, v in lean.items()} | {("fat", k): df[c] / h2 for k, c in _FAT.items()}
    pct = {key: col.groupby(df.sex).rank(pct=True) for key, col in columns.items()}

    def label(p: float) -> str:
        return "표준이하" if p <= 0.2 else "표준이상" if p > 0.8 else "표준"

    return [({k: label(pct[("lean", k)].iloc[i]) for k in SEGMENTS},
             {k: label(pct[("fat", k)].iloc[i]) for k in SEGMENTS}) for i in range(len(df))]


SEGMENTS = ("left_arm", "right_arm", "trunk", "left_leg", "right_leg")


def _cases() -> list[dict]:
    """KNHANES 2023에서 위험 유형별 규칙으로 5명을 고른다(규칙·순번 고정)."""
    import pandas as pd

    from app.services.prediction_service import predict_all
    from training.features import build_features, build_labels

    df = pd.read_sas("training/data/hn23_all.sas7bdat", encoding="latin1")
    X, Y = build_features(df), pd.DataFrame(build_labels(df))
    m = X.notna().all(axis=1) & df.age.between(25, 59) & df.BIA_FFM.notna() & df.HE_wt.notna()
    rows = []
    for i in df.index[m]:
        x = X.loc[i]
        prof = {"gender": "female" if x.female else "male", "birth_year": 2026 - int(x.age),
                **{k: bool(x[k]) for k in ("fh_hypertension", "fh_diabetes", "fh_dyslipidemia")}}
        inbody = {"weight": float(df.HE_wt[i]), "bmi": float(x.bmi), "body_fat_percentage": float(x.body_fat_percentage),
                  "body_fat_mass": float(df.BIA_BFM[i]), "body_water": float(df.BIA_TBW[i]),
                  "basal_metabolic_rate": round(370 + 21.6 * float(df.BIA_FFM[i]))}  # 인바디 기초대사량 식
        try:
            lv = {r["disease_key"]: r["risk_level"] for r in predict_all(inbody, prof)}
        except ValueError:
            continue
        actual = {k: (None if pd.isna(Y[k][i]) else int(Y[k][i])) for k in Y.columns}
        rows.append({"profile": prof, "inbody": inbody, "levels": lv, "actual": actual})

    def pick(name, cond, seed):
        found = [r for r in rows if cond(r)]
        return {"case": name, **found[seed % len(found)]}

    return [
        pick("고혈압 높음(실제 고혈압)", lambda r: r["levels"]["hypertension"] == "high"
             and r["actual"]["hypertension"] == 1, 3),
        pick("당뇨 높음(실제 당뇨)", lambda r: r["levels"]["diabetes"] == "high" and r["actual"]["diabetes"] == 1, 5),
        pick("여러 질환 높음, 비만", lambda r: sum(v == "high" for v in r["levels"].values()) >= 3
             and r["inbody"]["bmi"] >= 30, 1),
        pick("고중성지방 중간, 정상 BMI", lambda r: r["levels"]["triglyceride"] != "low" and r["inbody"]["bmi"] < 23
             and r["levels"]["hypertension"] == "low", 2),
        pick("위험 낮음(여성)", lambda r: all(v == "low" for v in r["levels"].values())
             and r["profile"]["gender"] == "female", 7),
    ]


def _synthetic_cases() -> list[dict]:
    """KNHANES 2023 타겟 연령(25–45세)에 합성 부위 판정을 붙여 부위 유형별로 5명을 고른다."""
    import pandas as pd

    from training.features import build_features

    df = pd.read_sas("training/data/hn23_all.sas7bdat", encoding="latin1")
    X = build_features(df)
    need = ["HE_ht", "HE_wt", "BIA_FFM", "BIA_BFM", "BIA_TBW", *_LEAN.values(), *_FAT.values()]
    df = df[X.notna().all(axis=1) & df.age.between(25, 45) & df[need].notna().all(axis=1)].reset_index(drop=True)
    X = build_features(df)
    labels = segmental_labels(df)
    rows = []
    for i, (lean, fat) in enumerate(labels):
        x = X.iloc[i]
        rows.append({
            "profile": {"gender": "female" if x.female else "male", "birth_year": 2026 - int(x.age),
                        **{k: bool(x[k]) for k in ("fh_hypertension", "fh_diabetes", "fh_dyslipidemia")}},
            "inbody": {"weight": float(df.HE_wt[i]), "height": float(df.HE_ht[i]), "bmi": float(x.bmi),
                       "body_fat_percentage": float(x.body_fat_percentage), "body_fat_mass": float(df.BIA_BFM[i]),
                       "body_water": float(df.BIA_TBW[i]), "basal_metabolic_rate": round(370 + 21.6 * float(df.BIA_FFM[i])),
                       "segmental_lean": lean, "segmental_fat": fat},
        })
    low = lambda r, k: r["inbody"]["segmental_lean"][k] == "표준이하"  # noqa: E731
    rules = [
        ("합성: 하체 근육 부족", lambda r: low(r, "left_leg") and low(r, "right_leg")),
        ("합성: 팔만 근육 부족", lambda r: low(r, "left_arm") and low(r, "right_arm")
         and not (low(r, "left_leg") or low(r, "right_leg"))),
        ("합성: 몸통 체지방 과다", lambda r: r["inbody"]["segmental_fat"]["trunk"] == "표준이상"
         and not (low(r, "left_leg") or low(r, "right_leg"))),
        ("합성: 다리 좌우 차이", lambda r: r["inbody"]["segmental_lean"]["left_leg"]
         != r["inbody"]["segmental_lean"]["right_leg"]),
        ("합성: 부위 모두 표준", lambda r: set(r["inbody"]["segmental_lean"].values()) == {"표준"}
         and set(r["inbody"]["segmental_fat"].values()) == {"표준"}),
    ]
    return [{"case": name, **next(r for r in rows if cond(r))} for name, cond in rules]


def with_muscle(cases: list[dict]) -> list[dict]:
    return [{**c, "inbody": {**c["inbody"], "muscle_control": (0.0, 2.0)[i % 2]}} for i, c in enumerate(cases)]


def with_previous(cases: list[dict]) -> list[dict]:
    out = []
    for i, case in enumerate(cases):
        d_bmi, d_pbf = _PREVIOUS_DELTAS[i % len(_PREVIOUS_DELTAS)]
        inbody = case["inbody"]
        previous = {**inbody, "measured_at": PREVIOUS_MEASURED_AT, "bmi": round(inbody["bmi"] + d_bmi, 1),
                    "body_fat_percentage": round(inbody["body_fat_percentage"] + d_pbf, 1)}
        out.append({**case, "previous": previous})
    return out


def case_risk_change(case: dict, risks: list[dict]) -> list[dict] | None:
    """앱의 리포트 생성과 같은 계산(risk_store 모양 → risk_change). DB는 쓰지 않는다."""
    from app.services import risk_change, risk_store

    if not case.get("previous") or not risks:
        return None
    before = risk_store.compute(case["previous"], case["profile"])
    if before is None:
        return None
    now = risk_store.from_predictions(case["inbody"], case["profile"], risks)
    return risk_change.compute(before, now, case["previous"]["measured_at"])


def _report(case: dict) -> dict:
    """앱의 _analyze_inbody_record와 같은 순서(DB·추세 제외)."""
    from app.services.ai_service import generate_report
    from app.services.citation_service import validate_citations
    from app.services.personal_metrics import compute
    from app.services.prediction_service import predict_all
    from app.services.rag_taxonomy import build_queries, trait_queries
    from app.services.retrieval_service import retrieve

    prof, inbody = case["profile"], case["inbody"]
    risks = predict_all(inbody, prof)
    metrics = compute(inbody, prof)
    traits = trait_queries(risks, prof, metrics, None)
    queries = [*build_queries(risks, inbody, prof["gender"]), *traits]
    evidence = retrieve(queries)
    changes = case_risk_change(case, risks)
    raw = generate_report(risks, prof, inbody_metrics=inbody, evidence=evidence, personal_metrics=metrics,
                          risk_change=changes)
    content, sources = validate_citations(raw, evidence)
    trait_hits = [{"trait": q.trait, "query": q.text,
                   "chunks": [f"{e.title} › {e.heading_path or ''}" for e in evidence if e.trait == q.trait]}
                  for q in traits]
    return {**{k: case[k] for k in case if k != "levels"}, "risks": risks, "personal_metrics": metrics,
            "risk_change": changes, "queries": [q.text for q in queries], "trait_hits": trait_hits, "report": content, "sources": sources}


def run(out: Path, inbody_json: Path | None, synthetic: bool = False, previous: bool = False,
        muscle: bool = False) -> None:
    from dotenv import load_dotenv

    load_dotenv(".env")
    cases = (_synthetic_cases() if synthetic else _cases()) + (
        json.loads(inbody_json.read_text(encoding="utf-8")) if inbody_json else [])
    if muscle:
        cases = with_muscle(cases)
    if previous:
        cases = with_previous(cases)
    results = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []  # 중간에 끊겨도 이어서 한다
    done = {r["case"] for r in results}
    out.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case["case"] in done:
            continue
        if results:
            time.sleep(15)  # Gemini 임베딩 무료 등급 분당 100건. 리포트 1건에 쿼리 약 20개
        results.append(_report(case))
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"{case['case']}: 쿼리 {len(results[-1]['queries'])}, 출처 {len(results[-1]['sources'])}")


def _cited(r: dict) -> set:
    return {(s["title"], s.get("page"), s.get("heading_path")) for s in r["sources"]}


def _metrics_of(r: dict) -> dict:
    if "personal_metrics" in r:
        return r["personal_metrics"]
    from app.services.personal_metrics import compute  # 바꾸기 전 결과에는 없으므로 같은 입력으로 계산한다

    return compute(r["inbody"], r["profile"])


def summarize(results: list[dict]) -> dict:
    actions = [(a, r) for r in results for p in r["report"]["lifestyle_plan"] for a in p["actions"]]
    numbered = sum(mentions_user_number(a, [r["inbody"], _metrics_of(r)]) for a, r in actions)
    missing = sum(item["metric"] in _metrics_of(r)["missing_metrics"] for r in results
                  for key in ("strengths", "watch_points") for item in r["report"]["body_composition_report"][key])
    return {"jaccard": round(jaccard_mean([_cited(r) for r in results]), 3),
            "numbered_actions": f"{numbered}/{len(actions)}", "missing_metric_items": missing}


def render(r: dict) -> str:
    """프론트 ReportScreen 섹션 순서·라벨대로 마크다운."""
    rep, out = r["report"], []
    rr = rep["risk_report"]
    out.append(f"### {r['case']}\n**종합 건강 요약** — 전반적 위험도 {BADGE[rr[0]['risk_level']] if rr else '-'} · "
               f"{len(rr)}개 지표\n> {rep['summary']['overall_message']}\n\n**세부 위험도 분석**")
    out += [f"- {x['disease']} [{BADGE[x['risk_level']]}] {x['probability'] * 100:.1f}% — {x['message']}" for x in rr]
    bc = rep["body_composition_report"]
    out.append("\n**인바디 해석**")
    out += [f"- {s['metric']}: {s['message']}" for s in bc["strengths"]]
    out += [f"- {w['metric']} [{BADGE[w['priority']]}] {w['current_value']} — {w['interpretation']}"
            for w in bc["watch_points"]]
    out.append("\n**맞춤형 개선 가이드**")
    for i, p in enumerate(rep["lifestyle_plan"], 1):
        cite = f" [{', '.join(p['source_ids'])}]" if p.get("source_ids") else ""
        out.append(f"{i}. **{CATEGORY.get(p['category'], p['category'])} · {p['title']}**{cite}\n   {p['reason']}")
        out += [f"   - {a}" for a in p["actions"]]
    if rep["supplement_candidates"]:
        out.append("\n**보충제 후보**")
        out += [f"- {s['name']}: {s['why_consider']} [{', '.join(s['source_ids'])}]" for s in rep["supplement_candidates"]]
    out.append(f"\n**주의사항** — {rep['cautions']['medical_disclaimer']}")
    if r["sources"]:
        out.append("<sub>출처: " + "; ".join(f"{s['sid']} {s['title']}" + (f" p.{s['page']}" if s.get("page") else "")
                                           for s in r["sources"]) + "</sub>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="리포트 미리보기·개인화 비교")
    sub = parser.add_subparsers(dest="action", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--inbody-json", type=Path)
    r.add_argument("--synthetic", action="store_true", help="KNHANES에 합성 부위 판정을 붙인 5명")
    r.add_argument("--with-previous", action="store_true", help="1년 전 합성 기록을 붙여 위험도 변화까지")
    r.add_argument("--with-muscle", action="store_true", help="합성 근육조절로 체성분 유형까지")
    c = sub.add_parser("compare")
    c.add_argument("before", type=Path)
    c.add_argument("after", type=Path)
    args = parser.parse_args(argv)
    if args.action == "run":
        run(args.out, args.inbody_json, args.synthetic, args.with_previous, args.with_muscle)
        return 0
    before, after = (json.loads(p.read_text(encoding="utf-8")) for p in (args.before, args.after))
    n = min(len(before), len(after))  # 바꾸기 전 결과는 KNHANES 5명뿐이다
    print(json.dumps({"before": summarize(before[:n]), "after": summarize(after[:n])}, ensure_ascii=False))
    for res in after:
        print("\n" + render(res))
        for c in res.get("risk_change") or []:
            print(f"  · 변화 [{c['disease']}] {c['previous_pct']}→{c['current_pct']}% ({c['change_pp']:+}%p) 몫 {c['share_pp']}")
        for t in res.get("trait_hits", []):
            print(f"  · 특성 [{t['trait']}] {t['query']} → {len(t['chunks'])}개 {t['chunks'][:2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
