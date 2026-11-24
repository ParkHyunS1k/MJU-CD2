import pytest

from rag import mfds


def _a(name, fn):  # I2710 품목분류정보 행
    return {"PRDCT_NM": name, "PRIMARY_FNCLTY": fn, "DAY_INTK_LOWLIMIT": "1.4", "DAY_INTK_HIGHLIMIT": "4.2",
            "INTK_UNIT": "g", "INTK_MEMO": "", "IFTKN_ATNT_MATR_CN": "주의", "LAST_UPDT_DTM": "20200820"}


def _b(name, no, fn):  # I-0050 개별인정형 행
    return {"RAWMTRL_NM": name, "HF_FNCLTY_MTRAL_RCOGN_NO": no, "PRIMARY_FNCLTY": fn,
            "DAY_INTK_LOWLIMIT": "500 (mg/일)", "DAY_INTK_HIGHLIMIT": "500 (mg/일)", "IFTKN_ATNT_MATR_CN": ""}


def test_tags_from_function_text():
    assert mfds.tags_of("식후 혈당상승 억제에 도움을 줄 수 있음") == ["blood_glucose"]
    assert mfds.tags_of("혈중 중성지질 개선·혈중 콜레스테롤 개선에 도움") == ["cholesterol", "triglyceride"]
    assert mfds.tags_of("(국문) 근력개선에 도움을 줄 수 있음") == ["muscle"]
    assert mfds.tags_of("관절 건강에 도움을 줄 수 있음") == []


def test_nutrient_muscle_function_is_not_muscle_tag():
    for text in ("신경과 근육 기능 유지에 필요", "근육, 결합조직 등 신체조직의 구성성분"):
        assert "muscle" not in mfds.tags_of(text)


def test_recognition_no_and_clean_name():
    assert mfds.recognition_no("Nopal 추출물(제2009-39호)") == "2009-39"
    assert mfds.recognition_no("차조기등복합추출물(기능성원료인정New제2007-13호)") == "2007-13"
    assert mfds.recognition_no("아라비아검(아카시아검)") is None
    assert mfds.clean_name("두충우슬추출복합물(KGC08EA)(제2023-40호)") == "두충우슬추출복합물(KGC08EA)"


def test_build_keeps_relevant_adds_missing_individual_and_formats_intake():
    i2710 = [_a("바나바잎 추출물", "식후 혈당상승 억제에 도움을 줄 수 있음"),
             _a("Nopal 추출물(제2009-39호)", "식후 혈당상승 억제에 도움을 줄 수 있음"),
             _a("루테인", "노화로 인해 감소될 수 있는 황반색소밀도를 유지")]
    i0050 = [_b("Nopal 추출물(기능성원료인정제2009-39호)", "제2009-39호", "혈당"),
             _b("아마인(기능성원료인정제2007-4호)", "제2007-4호", "혈중 콜레스테롤 개선에 도움")]
    out = mfds.build_ingredients(i2710, i0050)
    assert [i["name"] for i in out] == ["바나바잎 추출물", "Nopal 추출물", "아마인"]
    assert [i["notified"] for i in out] == [True, False, False]
    assert out[0]["intake"] == "1.4–4.2 g" and out[0]["updated"] == "20200820"
    assert out[2]["recognition_no"] == "2007-4" and out[2]["updated"] is None
    assert out[2]["intake"] == "500 (mg/일)" and out[2]["cautions"] is None


def test_count_check_refuses_empty_or_shrunk_lists():
    with pytest.raises(mfds.IncompleteFetch):
        mfds.check_counts([], previous=None)
    with pytest.raises(mfds.IncompleteFetch):
        mfds.check_counts([{}] * 70, previous=[{}] * 100)
    mfds.check_counts([{}] * 90, previous=[{}] * 100)


def test_intake_memo_is_not_shown():
    # 메모는 대부분 "(24.12.23.) 일일섭취량 현행화" 같은 변경 이력이라 섭취량 문장에 넣지 않는다
    row = {**_a("공액리놀레산", "체지방 감소에 도움을 줄 수 있음"), "INTK_MEMO": "(24.12.23.) 일일섭취량 현행화"}
    assert mfds.build_ingredients([row], [])[0]["intake"] == "1.4–4.2 g"
