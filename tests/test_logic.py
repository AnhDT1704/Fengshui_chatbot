"""
test_logic.py – Unit test cho LOGIC THUẦN (deterministic, không cần LLM/mạng).

Bao phủ đúng các lớp lỗi hay gặp: tính size sai cung, đọc số kho thô, sai mệnh,
CTA Shopee nuốt mất nội dung, capture metadata sai, parse JSON model lỗi.
"""

import json

import pytest

import fengshui_finetune_client as ffc
import graph
import knowledge_base_agent as kb
import memory
import skills_agent
from langchain_core.messages import ToolMessage


# ═══════════════════════ TÍNH SIZE VÒNG (skills_agent) ═══════════════════════

@pytest.mark.parametrize("wrist,li,count,length,cung", [
    (18, 6, 30, 18.0, "Lão"),
    (18, 8, 23, 18.4, "Bệnh"),
    (18, 10, 18, 18.0, "Lão"),
    (20, 6, 34, 20.4, "Lão"),
    (20, 8, 25, 20.0, "Sinh"),
    (20, 10, 21, 21.0, "Sinh"),
])
def test_compute_bracelet_recommended(wrist, li, count, length, cung):
    r = skills_agent.compute_bracelet(wrist, li)["recommended"]
    assert r["count"] == count
    assert r["length_cm"] == length
    assert r["fengshui"] == cung


def test_bracelet_math_invariant():
    """Mọi ứng viên: dài = số hạt × đường kính; cung = số hạt % 4."""
    diam = {6: 0.6, 8: 0.8, 10: 1.0}
    cung = ["Tử", "Sinh", "Lão", "Bệnh"]
    for wrist in (13, 14.5, 16, 17.5, 19, 21):
        for li in (6, 8, 10):
            res = skills_agent.compute_bracelet(wrist, li)
            for c in [res["recommended"], *res["alternatives"]]:
                assert c["length_cm"] == round(c["count"] * diam[li], 1)
                assert c["fengshui"] == cung[c["count"] % 4]
                assert c["length_cm"] >= wrist - 0.6  # không quá chật


def test_phong_thuy_cycle():
    assert skills_agent._phong_thuy(1) == ("Sinh", True)
    assert skills_agent._phong_thuy(2) == ("Lão", True)
    assert skills_agent._phong_thuy(3) == ("Bệnh", False)
    assert skills_agent._phong_thuy(4) == ("Tử", False)


def test_recommend_li():
    assert skills_agent.recommend_li(14) == 6
    assert skills_agent.recommend_li(17) == 8
    assert skills_agent.recommend_li(19) == 10


# ═══════════════════════ CÂU TỒN KHO (memory._stock_phrase) ═══════════════════

@pytest.mark.parametrize("in_stock,qty,expected", [
    (True, 939235, "còn nhiều hàng"),   # số kho khổng lồ → KHÔNG đọc số
    (True, 11, "còn nhiều hàng"),
    (True, 10, "còn 10 sản phẩm (sắp hết)"),
    (True, 3, "còn 3 sản phẩm (sắp hết)"),
    (True, 0, "còn hàng"),
    (True, None, "còn hàng"),
    (False, 5, "hết hàng"),
    (False, 0, "hết hàng"),
])
def test_stock_phrase(in_stock, qty, expected):
    assert memory._stock_phrase(in_stock, qty) == expected


# ═══════════════════════ MỆNH TỪ NĂM SINH (knowledge_base_agent) ══════════════

@pytest.mark.parametrize("year,element,can_chi", [
    (1924, "Kim", "Giáp Tý"),
    (1984, "Kim", "Giáp Tý"),
    (1990, "Thổ", "Canh Ngọ"),
    (2004, "Thủy", "Giáp Thân"),
])
def test_year_to_can_chi(year, element, can_chi):
    d = kb._year_to_can_chi(year)
    assert d["element"] == element
    assert d["can_chi"] == can_chi


def test_fengshui_from_code_shape():
    r = kb._fengshui_result_from_code(1990)
    assert r["element"] == "Thổ"
    assert r["lucky_colors"]
    assert r["suggested_filter_elements"]


# ═══════════════════════ SHOPEE CTA (chống cụt nội dung) ══════════════════════

def test_cta_keeps_long_promo_line(monkeypatch):
    """Regression: LLM viết KM + link chung 1 dòng → KHÔNG được nuốt mất KM."""
    monkeypatch.setattr(graph, "_wants_shopee_cta", lambda t: True)
    ans = ("Dạ ngày 8/8 sắp tới shop có Sale 8/8 giảm 13% cho mọi sản phẩm, bạn đón xem "
           "tại [sàn Shopee của shop](https://shopee.vn/vananhome?x=1) nhé ạ.")
    out = graph._apply_shopee_cta("có giảm giá gì", ans)
    assert "8/8" in out and "13%" in out                 # nội dung KM còn nguyên
    assert out.rstrip().endswith(graph.SHOPEE_CTA)        # có CTA chuẩn ở cuối
    assert out.count("shopee.vn/vananhome") == 1          # đúng 1 link, không lặp


def test_cta_drops_standalone_link_line(monkeypatch):
    monkeypatch.setattr(graph, "_wants_shopee_cta", lambda t: True)
    ans = "Dạ còn hàng ạ.\n[Shopee](https://shopee.vn/vananhome)"
    out = graph._apply_shopee_cta("còn hàng không", ans)
    assert "Dạ còn hàng ạ." in out
    assert out.count("shopee.vn/vananhome") == 1


def test_cta_absent_when_not_wanted(monkeypatch):
    monkeypatch.setattr(graph, "_wants_shopee_cta", lambda t: False)
    ans = "Dạ đá này ý nghĩa bình an ạ. [Shopee](https://shopee.vn/vananhome)"
    out = graph._apply_shopee_cta("ý nghĩa gì", ans)
    assert "shopee.vn/vananhome" not in out


# ═══════════════════════ CAPTURE METADATA (graph._focused_elements) ═══════════

def _tool_msg(payload):
    return ToolMessage(content=json.dumps(payload, ensure_ascii=False), tool_call_id="t")


def test_focused_elements_multi_person():
    msgs = [
        _tool_msg({"element": "Thủy", "birth_year": 2004, "can_chi": "Giáp Thân"}),
        _tool_msg({"element": "Thổ", "birth_year": 1990, "can_chi": "Canh Ngọ"}),
    ]
    out = graph._focused_elements(msgs)
    assert {e["element"] for e in out} == {"Thủy", "Thổ"}
    assert all(e["certainty"] == "confirmed" for e in out)


def test_focused_elements_skip_need_more_info():
    assert graph._focused_elements([_tool_msg({"need_more_info": True, "chi": "Ngọ"})]) is None


def test_focused_elements_none_when_no_menh():
    assert graph._focused_elements([_tool_msg({"name": "vòng tay", "category": "vòng tay"})]) is None


# ═══════════════════════ PARSE JSON MODEL FT (fengshui_finetune_client) ═══════

def test_parse_json_blob_strips_think():
    assert ffc.parse_json_blob('<think>tính toán</think>\n{"element":"Thủy"}') == {"element": "Thủy"}


def test_parse_json_blob_plain():
    assert ffc.parse_json_blob('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_blob_bad_returns_raw():
    assert "raw" in ffc.parse_json_blob("không có json ở đây")
