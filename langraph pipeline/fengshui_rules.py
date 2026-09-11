# -*- coding: utf-8 -*-
"""
fengshui_rules.py – Rule CODE deterministic: màu ↔ mệnh ↔ năm.

Dùng SAU khi model PT trả mệnh (năm→mệnh), hoặc khi có colors SP từ VLM/DB.
Không thay model finetune — chỉ bổ sung lucky/unlucky colors + match SP + năm.

Public API:
  normalize_element, menh_to_colors, colors_to_elements,
  match_user_menh_product_colors, enrich_menh_payload,
  years_for_element, elements_to_years
"""
from __future__ import annotations

import re
from typing import Any, Optional

# ── Ngũ hành quan hệ + màu (khớp knowledge_base_agent.ELEMENT_INFO) ──
ELEMENTS = ["Kim", "Mộc", "Thủy", "Hỏa", "Thổ"]

# Kim sinh Thủy, Thủy sinh Mộc, Mộc sinh Hỏa, Hỏa sinh Thổ, Thổ sinh Kim
SINH = {"Kim": "Thủy", "Thủy": "Mộc", "Mộc": "Hỏa", "Hỏa": "Thổ", "Thổ": "Kim"}
# Kim khắc Mộc, Mộc khắc Thổ, Thổ khắc Thủy, Thủy khắc Hỏa, Hỏa khắc Kim
KE = {"Kim": "Mộc", "Mộc": "Thổ", "Thổ": "Thủy", "Thủy": "Hỏa", "Hỏa": "Kim"}

ELEMENT_INFO = {
    "Kim": {
        "generating_element": "Thổ",
        "controlling_element": "Hỏa",
        "lucky_color_groups": [
            {"colors": ["vàng", "nâu"], "reason": "Thổ sinh Kim (tương sinh)"},
            {"colors": ["trắng", "bạc", "xám"], "reason": "bản mệnh Kim"},
        ],
        "unlucky_colors": ["đỏ", "hồng", "tím", "cam"],
    },
    "Mộc": {
        "generating_element": "Thủy",
        "controlling_element": "Kim",
        "lucky_color_groups": [
            {"colors": ["đen", "xanh dương", "xanh nước"], "reason": "Thủy sinh Mộc"},
            {"colors": ["xanh lá", "xanh rêu", "nâu"], "reason": "bản mệnh / gỗ Mộc"},
        ],
        "unlucky_colors": ["trắng", "bạc", "xám"],
    },
    "Thủy": {
        "generating_element": "Kim",
        "controlling_element": "Thổ",
        "lucky_color_groups": [
            {"colors": ["trắng", "bạc"], "reason": "Kim sinh Thủy"},
            {"colors": ["đen", "xanh dương", "xanh nước"], "reason": "bản mệnh Thủy"},
        ],
        "unlucky_colors": ["vàng", "nâu", "đỏ", "cam"],
    },
    "Hỏa": {
        "generating_element": "Mộc",
        "controlling_element": "Thủy",
        "lucky_color_groups": [
            {"colors": ["xanh lá", "xanh rêu"], "reason": "Mộc sinh Hỏa"},
            {"colors": ["đỏ", "hồng", "tím", "cam"], "reason": "bản mệnh Hỏa"},
        ],
        "unlucky_colors": ["đen", "xanh dương", "xám"],
    },
    "Thổ": {
        "generating_element": "Hỏa",
        "controlling_element": "Mộc",
        "lucky_color_groups": [
            {"colors": ["đỏ", "hồng", "tím", "cam"], "reason": "Hỏa sinh Thổ"},
            {"colors": ["vàng", "nâu", "be"], "reason": "bản mệnh Thổ"},
        ],
        "unlucky_colors": ["xanh lá", "xanh rêu", "xanh dương"],
    },
}

# Màu → hành (chuẩn hoá token)
COLOR_TO_ELEMENTS: dict[str, list[str]] = {
    "trắng": ["Kim"],
    "bạc": ["Kim"],
    "xám": ["Kim"],
    "vàng": ["Thổ", "Kim"],
    "nâu": ["Thổ", "Mộc"],
    "be": ["Thổ"],
    "đen": ["Thủy"],
    "xanh dương": ["Thủy"],
    "xanh nước": ["Thủy"],
    "xanh da trời": ["Thủy"],
    "xanh lá": ["Mộc"],
    "xanh rêu": ["Mộc"],
    "xanh lục": ["Mộc"],
    "đỏ": ["Hỏa"],
    "hồng": ["Hỏa"],
    "cam": ["Hỏa"],
    "tím": ["Hỏa"],
    "đa sắc": ["Kim", "Mộc", "Thủy", "Hỏa", "Thổ"],
}

_COLOR_ALIASES = {
    "xanh lục": "xanh lá",
    "màu gỗ trầm": "nâu",
    "gỗ trầm": "nâu",
    "xanh biển": "xanh dương",
    "xanh nước biển": "xanh dương",
    "kem": "be",
}

CAN = ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"]
CHI = ["Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi", "Thân", "Dậu", "Tuất", "Hợi"]


def normalize_element(element: Any) -> Optional[str]:
    if element is None:
        return None
    e = str(element).replace("Mệnh", "").replace("mệnh", "").strip()
    for cand in ELEMENTS:
        if e.lower() == cand.lower():
            return cand
    return None


def _norm_color_token(tok: str) -> Optional[str]:
    c = re.sub(r"\s+", " ", str(tok)).strip().lower()
    if not c or c in {"không xác định", "n/a", "none", "null"}:
        return None
    c = _COLOR_ALIASES.get(c, c)
    return c


def normalize_colors(colors: Any) -> list[str]:
    """Tách list/string màu → token chuẩn, unique."""
    out: list[str] = []
    if colors is None:
        return out
    items = [colors] if isinstance(colors, str) else list(colors)
    for raw in items:
        if raw is None:
            continue
        for tok in re.split(r"[,+/]| và | & ", str(raw), flags=re.I):
            c = _norm_color_token(tok)
            if c and c not in out:
                out.append(c)
    return out


def menh_to_colors(element: str) -> dict:
    """Mệnh → màu hợp / kỵ + hành sinh/khắc."""
    e = normalize_element(element)
    if not e:
        return {"error": f"Không nhận ra mệnh {element!r}. Hợp lệ: {ELEMENTS}"}
    rel = ELEMENT_INFO[e]
    lucky = []
    for g in rel["lucky_color_groups"]:
        for c in g["colors"]:
            if c not in lucky:
                lucky.append(c)
    return {
        "element": e,
        "generating_element": rel["generating_element"],
        "controlling_element": rel["controlling_element"],
        "lucky_color_groups": rel["lucky_color_groups"],
        "lucky_colors": lucky,
        "unlucky_colors": list(rel["unlucky_colors"]),
        "suggested_filter_elements": [e, rel["generating_element"]],
        "source": "fengshui_rules",
    }


def colors_to_elements(colors: Any) -> dict:
    """Colors → các hành gắn với màu (union) qua bảng COLOR_TO_ELEMENTS.

    Lưu ý: bảng này là map màu→hành trực tiếp (vd đỏ→Hỏa). Để lấy ĐỦ mệnh
    coi màu đó là màu hợp (gồm tương sinh, vd đỏ → Hỏa + Thổ), dùng
    colors_to_lucky_menh().
    """
    cols = normalize_colors(colors)
    found: list[str] = []
    detail = []
    for c in cols:
        els = COLOR_TO_ELEMENTS.get(c)
        if not els:
            # fuzzy contains
            for k, v in COLOR_TO_ELEMENTS.items():
                if k in c or c in k:
                    els = v
                    break
        if els:
            detail.append({"color": c, "elements": els})
            for e in els:
                if e not in found:
                    found.append(e)
    return {
        "colors": cols,
        "elements": found,
        "detail": detail,
        "all_five": set(found) >= set(ELEMENTS),
        "source": "fengshui_rules",
    }


def colors_to_lucky_menh(colors: Any) -> dict:
    """Colors → mọi mệnh coi màu đó là màu HỢP (bản mệnh hoặc tương sinh).

    Ví dụ: đỏ ∈ màu hợp của Hỏa (bản mệnh) và Thổ (Hỏa sinh Thổ)
    → trả ['Hỏa', 'Thổ'], không chỉ ['Hỏa'].
    Đa sắc → cả 5 hành.
    """
    cols = normalize_colors(colors)
    if not cols:
        return {
            "colors": [],
            "elements": [],
            "detail": [],
            "all_five": False,
            "source": "fengshui_rules",
        }
    if "đa sắc" in cols:
        return {
            "colors": cols,
            "elements": list(ELEMENTS),
            "detail": [{"color": "đa sắc", "elements": list(ELEMENTS)}],
            "all_five": True,
            "source": "fengshui_rules",
        }

    pset = set(cols)
    found: list[str] = []
    detail = []
    for e in ELEMENTS:
        info = menh_to_colors(e)
        lucky = set(normalize_colors(info.get("lucky_colors") or []))
        hit = sorted(pset & lucky)
        if hit:
            found.append(e)
            detail.append({"element": e, "lucky_hit": hit})

    return {
        "colors": cols,
        "elements": found,
        "detail": detail,
        "all_five": set(found) >= set(ELEMENTS),
        "source": "fengshui_rules",
    }


def match_user_menh_product_colors(
    user_element: str,
    product_colors: Any,
    *,
    product_compatible_elements: Any = None,
) -> dict:
    """
    So mệnh user với màu SP (và optional compatible_elements DB).

    verdict: hop | ky | trung
    """
    e = normalize_element(user_element)
    if not e:
        return {"error": f"Không nhận ra mệnh {user_element!r}", "verdict": "trung"}

    info = menh_to_colors(e)
    lucky = set(normalize_colors(info["lucky_colors"]))
    unlucky = set(normalize_colors(info["unlucky_colors"]))
    pcols = normalize_colors(product_colors)
    pset = set(pcols)

    # Đa sắc / phủ 5 hành
    if "đa sắc" in pset or (product_compatible_elements and
                            set(normalize_element(x) or "" for x in (product_compatible_elements or []))
                            >= set(ELEMENTS)):
        return {
            "verdict": "hop",
            "strength": "all_elements",
            "user_element": e,
            "product_colors": pcols,
            "lucky_hit": sorted(lucky & pset),
            "unlucky_hit": [],
            "reason": "SP đa sắc / hợp mọi mệnh.",
            "lucky_colors": info["lucky_colors"],
            "unlucky_colors": info["unlucky_colors"],
            "source": "fengshui_rules",
        }

    lucky_hit = sorted(lucky & pset)
    unlucky_hit = sorted(unlucky & pset)

    # compatible_elements DB
    gen_ok = False
    if product_compatible_elements:
        pe = [normalize_element(x) for x in product_compatible_elements]
        pe = [x for x in pe if x]
        if e in pe or info["generating_element"] in pe:
            gen_ok = True
        if set(pe) >= set(ELEMENTS):
            gen_ok = True

    if unlucky_hit and not lucky_hit and not gen_ok:
        verdict, strength = "ky", "color_conflict"
        reason = f"Màu SP {unlucky_hit} thuộc nhóm kỵ mệnh {e}."
    elif lucky_hit or gen_ok:
        verdict, strength = "hop", "color_or_element_match"
        reason = (
            f"Màu hợp {lucky_hit}" if lucky_hit else
            f"compatible_elements DB chứa mệnh {e} / mẹ sinh."
        )
    else:
        verdict, strength = "trung", "no_strong_signal"
        reason = "Không thấy màu hợp rõ cũng không thấy màu kỵ rõ."

    return {
        "verdict": verdict,
        "strength": strength,
        "user_element": e,
        "product_colors": pcols,
        "lucky_hit": lucky_hit,
        "unlucky_hit": unlucky_hit,
        "reason": reason,
        "lucky_colors": info["lucky_colors"],
        "unlucky_colors": info["unlucky_colors"],
        "suggested_filter_elements": info["suggested_filter_elements"],
        "source": "fengshui_rules",
    }


def enrich_menh_payload(data: dict) -> dict:
    """
    Sau khi PT model trả JSON (year_to_menh / element): gắn màu hợp/kỵ bằng CODE.
    Giữ nguyên field model; bổ sung lucky_* / unlucky_* / suggested_filter_elements.
    """
    if not data or data.get("need_more_info"):
        return data
    if data.get("error") or data.get("raw"):
        return data

    element = data.get("element") or (
        str(data.get("personal_element") or "").replace("Mệnh", "").strip()
    )
    info = menh_to_colors(element)
    if info.get("error"):
        out = dict(data)
        out["color_enrich_error"] = info["error"]
        return out

    out = dict(data)
    out["element"] = info["element"]
    out.setdefault("generating_element", info["generating_element"])
    out.setdefault("controlling_element", info["controlling_element"])
    # Luôn ưu tiên rule code cho màu (model không còn train màu)
    out["lucky_color_groups"] = info["lucky_color_groups"]
    out["lucky_colors"] = info["lucky_colors"]
    out["unlucky_colors"] = info["unlucky_colors"]
    out["suggested_filter_elements"] = info["suggested_filter_elements"]
    out["color_source"] = "fengshui_rules"
    return out


def years_for_element(
    element: str,
    *,
    modern_min: int | None = None,
    modern_max: int | None = None,
    around_year: int | None = None,
    modern_n: int = 12,
) -> dict:
    """Mệnh → năm trong chu kỳ 60 (1924–1983) + ví dụ năm hiện đại — CODE thuần.

    example_years_modern ưu tiên các năm GẦN `around_year` (mặc định năm hiện tại),
    quá khứ gần trước, rồi tương lai gần — không cắt cứng ở 2017.
    """
    from datetime import datetime

    e = normalize_element(element)
    if not e:
        return {"error": f"Không nhận ra mệnh {element!r}"}

    now_y = int(around_year or datetime.now().year)
    if modern_max is None:
        modern_max = now_y + 2
    if modern_min is None:
        modern_min = now_y - 90

    # Late import tránh circular nặng khi load KB
    try:
        import knowledge_base_agent as kb
        year_fn = kb._year_to_can_chi
    except Exception:
        return {"error": "Không load được _year_to_can_chi"}

    years, canchis = [], []
    for y in range(1924, 1984):
        info = year_fn(y)
        if info["element"] == e:
            years.append(y)
            canchis.append(info["can_chi"])

    modern: list[int] = []
    for y in years:
        # +60/+120/+180 để phủ gần hiện tại (vd 1964 → 2024)
        for add in (60, 120, 180):
            yy = y + add
            if modern_min <= yy <= modern_max:
                modern.append(yy)
    modern = sorted(set(modern))

    # Ưu tiên năm <= now (gần hiện tại nhất trước), rồi năm tương lai gần
    past = sorted([y for y in modern if y <= now_y], reverse=True)
    future = sorted([y for y in modern if y > now_y])
    example = (past + future)[: max(1, int(modern_n))]

    return {
        "element": e,
        "direction": "menh_to_years",
        "years_in_cycle": years,
        "can_chi_list": canchis,
        "example_years_modern": example,
        "count_in_cycle": len(years),
        "source": "fengshui_rules_code",
    }


def refresh_example_years_modern(data: dict, *, around_year: int | None = None, modern_n: int = 12) -> dict:
    """Ghi đè example_years_modern bằng CODE gần hiện tại (FT hay cắt list cũ)."""
    if not isinstance(data, dict):
        return data
    e = data.get("element")
    if not e:
        return data
    coded = years_for_element(e, around_year=around_year, modern_n=modern_n)
    if coded.get("error"):
        return data
    out = dict(data)
    out["example_years_modern"] = coded.get("example_years_modern") or []
    # Giữ years_in_cycle từ FT nếu đủ; nếu thiếu thì lấy code
    yc = out.get("years_in_cycle")
    if not isinstance(yc, list) or len(yc) < 8:
        out["years_in_cycle"] = coded.get("years_in_cycle") or yc
        out["can_chi_list"] = coded.get("can_chi_list") or out.get("can_chi_list")
        out["count_in_cycle"] = coded.get("count_in_cycle") or out.get("count_in_cycle")
    return out


def product_colors_to_compatible_years(product_colors: Any, **kwargs) -> dict:
    """Colors SP → elements → union năm hợp (code)."""
    ce = colors_to_elements(product_colors)
    if not ce["elements"]:
        return {
            "colors": ce["colors"],
            "elements": [],
            "years": [],
            "note": "Không map được màu → hành.",
            "source": "fengshui_rules",
        }
    all_years: list[int] = []
    per_el = []
    for e in ce["elements"]:
        yinfo = years_for_element(e, **kwargs)
        per_el.append(yinfo)
        if yinfo.get("example_years_modern"):
            all_years.extend(yinfo["example_years_modern"])
        elif yinfo.get("years_in_cycle"):
            all_years.extend(yinfo["years_in_cycle"])
    return {
        "colors": ce["colors"],
        "elements": ce["elements"],
        "example_years_modern": sorted(set(all_years)),
        "per_element": per_el,
        "source": "fengshui_rules",
    }
