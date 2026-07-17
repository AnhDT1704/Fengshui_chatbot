"""
build_fengshui_dataset.py – Sinh dataset TEXT→TEXT để finetune phần TRI THỨC PHONG THỦY.

Hai kỹ năng:
  A. Năm sinh → Can Chi → Nạp âm → Mệnh → màu hợp / kỵ
  B. Cổ tay (cm) → size hạt (li) → số hạt → chiều dài → cung Sinh-Lão-Bệnh-Tử

NGUỒN NHÃN: chính CODE đang chạy trong chatbot (knowledge_base_agent._year_to_can_chi
và skills_agent.compute_bracelet). Code là công thức chính xác → nhãn đúng 100%, sinh
được vô hạn mẫu, không phải gán tay.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
THIẾT KẾ SPLIT — ĐIỂM MẤU CHỐT CỦA THÍ NGHIỆM

Câu hỏi nghiên cứu: model HỌC ĐƯỢC QUY LUẬT hay chỉ HỌC THUỘC?

Nên KHÔNG chia ngẫu nhiên theo mẫu (làm vậy thì cùng một năm xuất hiện ở cả train lẫn
test → model chỉ cần nhớ, và ta đo được một con số đẹp nhưng VÔ NGHĨA).

Thay vào đó chia theo ĐƠN VỊ TRI THỨC:
  • NĂM  : mỗi năm chỉ thuộc DUY NHẤT một split. Test toàn năm model CHƯA TỪNG THẤY
           → đo được nó có nội suy ra quy luật chu kỳ 60 năm hay không.
  • CỔ TAY: train dùng bước 0.5cm (16.0, 16.5...); valid/test dùng giá trị XEN GIỮA
           (16.2, 16.7...) → đo khả năng nội suy trên miền liên tục.

Nếu model đúng cao trên các năm chưa thấy → nó thực sự học được luật (kết quả mạnh).
Nếu chỉ đúng trên năm đã thấy → nó học vẹt (cũng là kết quả đáng báo cáo).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CHẠY:
    python build_fengshui_dataset.py
    → dataset_fengshui/{train,valid,test}.jsonl   (định dạng messages, hợp Unsloth)
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "langraph pipeline"))

import knowledge_base_agent as kb   # noqa: E402  — nguồn nhãn cho phần MỆNH
import skills_agent as sk           # noqa: E402  — nguồn nhãn cho phần SIZE

OUT_DIR = Path("dataset_fengshui")
SEED = 42
YEAR_MIN, YEAR_MAX = 1900, 2100

SYSTEM_MESSAGE = (
    "Bạn là chuyên gia phong thủy của shop Vạn An Group. Trả lời các câu hỏi:\n"
    "1) NĂM SINH → tính Can Chi, Nạp âm, Mệnh ngũ hành, màu hợp và màu kỵ.\n"
    "2) CAN CHI đầy đủ (vd 'Canh Ngọ') → tính Nạp âm, Mệnh, màu hợp / kỵ.\n"
    "3) CHỈ CÓ CON GIÁP (vd 'tuổi Ngọ') → KHÔNG đủ dữ kiện: cùng một con giáp có 5 mệnh "
    "khác nhau theo chu kỳ 60 năm → phải HỎI LẠI năm sinh, TUYỆT ĐỐI không đoán mệnh.\n"
    "4) CỔ TAY (cm) → tính size hạt (li), số hạt, chiều dài vòng, cung Sinh-Lão-Bệnh-Tử.\n"
    "CHỈ trả về JSON, không giải thích thêm."
)

# Con giáp lặp mỗi 12 năm, mệnh theo chu kỳ 60 → mỗi con giáp đi qua ĐÚNG 5 mệnh.
CANCHI_PROMPTS = [
    "Tôi tuổi {cc}, mệnh gì vậy?",
    "{cc} thì thuộc mệnh nào ạ?",
    "Cho hỏi tuổi {cc} hợp màu gì?",
    "Mình {cc}, nên đeo vòng màu nào?",
]

# CHỈ có con giáp → THIẾU dữ kiện → phải hỏi lại năm sinh.
ZODIAC_PROMPTS = [
    "Tôi tuổi {chi}, mệnh gì vậy?",
    "Tuổi {chi} hợp màu nào ạ?",
    "Mình tuổi {chi}, nên mua vòng đá màu gì?",
    "Con giáp {chi} thuộc mệnh gì thế shop?",
    "Em tuổi {chi}, tư vấn màu hợp phong thủy giúp em.",
]

# Nhiều cách khách hỏi thật → model không bám vào một khuôn câu chữ.
YEAR_PROMPTS = [
    "Tôi sinh năm {y}, mệnh gì vậy?",
    "Sinh năm {y} thì thuộc mệnh nào ạ?",
    "Năm {y} là tuổi con giáp gì, mệnh gì?",
    "Cho hỏi tuổi {y} hợp màu nào?",
    "Mình {y}, nên đeo vòng đá màu gì cho hợp mệnh?",
    "Sinh {y} kỵ màu gì thế shop?",
    "{y} mệnh gì, nạp âm là gì?",
    "Em tuổi {y}, tư vấn giúp em màu hợp phong thủy với ạ.",
    "Bạn tôi sinh năm {y}, muốn mua vòng hợp mệnh thì chọn màu nào?",
    "Tuổi {y} hợp và kỵ những màu nào?",
]

WRIST_PROMPTS = [
    "Cổ tay tôi {w}cm thì đeo vòng bao nhiêu hạt?",
    "Tay mình {w} phân, chọn size hạt mấy li ạ?",
    "Chu vi cổ tay {w}cm nên mua vòng bao nhiêu hạt?",
    "Đo cổ tay được {w}cm, shop tư vấn size giúp mình.",
    "Cổ tay {w}cm thì vòng dài bao nhiêu, mấy hạt?",
    "Mình {w}cm cổ tay, muốn biết số hạt hợp phong thủy.",
]

WRIST_LI_PROMPTS = [
    "Cổ tay {w}cm mà mình thích hạt {li} li thì bao nhiêu hạt?",
    "Mình muốn size {li} li, cổ tay {w}cm, xâu mấy hạt ạ?",
    "Tay {w}cm, chọn {li} li thì vòng dài bao nhiêu?",
]


# ═══════════════════════════════════════════════════════════════════
#  ĐÁP ÁN (lấy từ CODE đang chạy — nguồn sự thật)
# ═══════════════════════════════════════════════════════════════════

def answer_year(year: int) -> dict:
    info = kb._year_to_can_chi(year)
    rel = kb.ELEMENT_INFO[info["element"]]
    lucky = [c for g in rel["lucky_color_groups"] for c in g["colors"]]
    return {
        "task":                "menh",
        "birth_year":          year,
        "can_chi":             info["can_chi"],
        "napam":               info["napam"],
        "element":             info["element"],
        "generating_element":  rel["generating_element"],   # hành SINH ra mệnh này
        "controlling_element": rel["controlling_element"],  # hành KHẮC mệnh này
        "lucky_colors":        lucky,
        "unlucky_colors":      rel["unlucky_colors"],
    }


def answer_canchi(can_chi: str) -> dict:
    """Can Chi ĐẦY ĐỦ (vd 'Canh Ngọ') → xác định DUY NHẤT một mệnh."""
    year = next(y for y in range(1924, 1984)
                if kb._year_to_can_chi(y)["can_chi"] == can_chi)
    a = answer_year(year)
    a.pop("birth_year")
    # Can Chi lặp mỗi 60 năm → không suy ra được 1 năm duy nhất, chỉ ra danh sách.
    a["example_years"] = [year - 60, year, year + 60]
    return a


def answer_zodiac_only(chi: str) -> dict:
    """CHỈ có con giáp → THIẾU dữ kiện. Model phải HỎI LẠI, không được đoán.

    Đây là mẫu quan trọng nhất của dataset: dạy model nhận ra giới hạn của chính nó.
    Một model chỉ biết trả lời mà không biết nói 'tôi chưa đủ dữ kiện' sẽ bốc đại 1
    trong 5 mệnh rồi tư vấn sai màu cho khách.
    """
    rows = [kb._year_to_can_chi(y) for y in range(1924, 1984)]
    same = [r for r in rows if r["chi"] == chi]
    return {
        "task":              "menh",
        "need_more_info":    True,
        "chi":               chi,
        "reason":            (f"Tuổi {chi} có 5 mệnh khác nhau tùy năm sinh "
                             f"(chu kỳ 60 năm), không thể xác định mệnh nếu chỉ biết con giáp."),
        "possible_can_chi":  [r["can_chi"] for r in same],
        "possible_elements": sorted({r["element"] for r in same}),
        "ask":               "Bạn cho shop xin NĂM SINH dương lịch để xác định đúng mệnh nhé!",
    }


def answer_wrist(wrist_cm: float, li: int | None = None) -> dict:
    chosen_li = li if li is not None else sk.recommend_li(wrist_cm)
    r = sk.compute_bracelet(wrist_cm, chosen_li)
    rec = r["recommended"]
    return {
        "task":            "size",
        "wrist_cm":        round(wrist_cm, 1),
        "bead_size_li":    chosen_li,
        "natural_li":      sk.recommend_li(wrist_cm),   # size shop tự đề xuất
        "bead_count":      rec["count"],
        "length_cm":       rec["length_cm"],
        "slack_cm":        rec["diff_cm"],              # dư so với cổ tay
        "fengshui":        rec["fengshui"],             # Sinh | Lão | Bệnh | Tử
        "is_fengshui_good": rec["is_fengshui"],
        # False = đã phải HY SINH phong thủy để vòng vừa tay (luật đánh đổi của shop)
        "fengshui_fits":   r["fengshui_fits"],
    }


# ═══════════════════════════════════════════════════════════════════
#  CHUỖI SUY LUẬN (<think>) — biến thể B của thí nghiệm
# ═══════════════════════════════════════════════════════════════════
# Biến thể A chỉ đưa ĐÁP ÁN → model có thể học vẹt (nhớ 140 cặp năm→mệnh).
# Biến thể B bắt model VIẾT RA PHÉP TÍNH trước khi trả lời → nó buộc phải THỰC HIỆN
# (năm−1924)%60 chứ không tra trí nhớ.
#
# LƯU Ý TRUNG THỰC: thêm <think> KHÔNG đảm bảo model học được quy luật — nó chỉ TĂNG CƠ
# HỘI. Đó chính là thứ tập test_extrapolate dùng để ĐO, chứ không phải để hy vọng.

def think_year(year: int) -> str:
    off = (year - 1924) % 60
    i = kb._year_to_can_chi(year)
    rel = kb.ELEMENT_INFO[i["element"]]
    lucky = [c for g in rel["lucky_color_groups"] for c in g["colors"]]
    return "\n".join([
        "Bước 1 — Vị trí trong chu kỳ 60 năm (mốc 1924 = Giáp Tý):",
        f"  offset = ({year} - 1924) % 60 = {year - 1924} % 60 = {off}",
        f"Bước 2 — Can (10 can): CAN[{off} % 10] = CAN[{off % 10}] = {i['can']}",
        f"Bước 3 — Chi (12 chi): CHI[{off} % 12] = CHI[{off % 12}] = {i['chi']}",
        f"  => Can Chi = {i['can_chi']}",
        "Bước 4 — Nạp âm (mỗi nạp âm ứng 2 năm liên tiếp):",
        f"  NAPAM[{off} // 2] = NAPAM[{off // 2}] = {i['napam']}  => mệnh {i['element']}",
        f"Bước 5 — Ngũ hành của mệnh {i['element']}:",
        f"  {rel['generating_element']} sinh {i['element']} (tương sinh, đại cát); "
        f"{rel['controlling_element']} khắc {i['element']} (đại kỵ)",
        f"  => màu hợp: {', '.join(lucky)}",
        f"  => màu kỵ: {', '.join(rel['unlucky_colors'])}",
    ])


def think_zodiac(chi: str) -> str:
    rows = [kb._year_to_can_chi(y) for y in range(1924, 1984)]
    same = [r for r in rows if r["chi"] == chi]
    lines = [
        f"Bước 1 — Khách chỉ cho CON GIÁP ({chi}), không cho năm sinh.",
        "Bước 2 — Con giáp lặp mỗi 12 năm, nhưng mệnh theo chu kỳ 60 năm.",
        f"  => trong 1 chu kỳ, tuổi {chi} xuất hiện 60/12 = 5 lần, mỗi lần một Nạp âm khác:",
    ]
    lines += [f"    {r['can_chi']:<10} - {r['napam']:<16} -> mệnh {r['element']}" for r in same]
    lines += [
        f"Bước 3 — 5 khả năng cho ra {len({r['element'] for r in same})} mệnh khác nhau "
        f"=> KHÔNG thể xác định mệnh.",
        "Bước 4 — Phải HỎI LẠI năm sinh, tuyệt đối không đoán bừa.",
    ]
    return "\n".join(lines)


def think_wrist(wrist_cm: float, li: int | None = None) -> str:
    chosen = li if li is not None else sk.recommend_li(wrist_cm)
    d = sk.BEAD_DIAM_CM[chosen]
    r = sk.compute_bracelet(wrist_cm, chosen)
    rec = r["recommended"]
    ideal = wrist_cm + sk.TARGET_SLACK

    lines = []
    if li is None:
        lines += [
            f"Bước 1 — Chọn size hạt theo cổ tay {wrist_cm}cm "
            f"(<=15.9 -> 6 li; <18 -> 8 li; >=18 -> 10 li)  => {chosen} li "
            f"(đường kính {d}cm)",
        ]
    else:
        lines += [
            f"Bước 1 — Khách CHỈ ĐỊNH {chosen} li (đường kính {d}cm). "
            f"Size tự nhiên theo cổ tay là {sk.recommend_li(wrist_cm)} li.",
        ]
    lines += [f"Bước 2 — Chiều dài mục tiêu = cổ tay + {sk.TARGET_SLACK} = {round(ideal, 2)}cm"]

    # Liệt kê các số hạt vừa tay + cung phong thủy của từng phương án.
    lines.append("Bước 3 — Thử các số hạt (dài = số hạt x đường kính; cung = số hạt % 4):")
    lo = max(1, int((wrist_cm + sk.GEN_MIN) / d))
    for n in range(lo, lo + 6):
        length = round(n * d, 1)
        diff = round(length - wrist_cm, 1)
        lab, good = sk._phong_thuy(n)
        verdict = "loại (ngắn hơn cổ tay)" if diff < sk.REC_MIN else (
            "loại (rộng quá)" if diff > sk.REC_MAX else f"vừa tay, dư {diff}cm")
        lines.append(f"    {n:>2} hạt = {length}cm | {n} % 4 = {n % 4} -> {lab}"
                     f"{' (tốt)' if good else ' (xấu)'} | {verdict}")

    lines += [
        f"Bước 4 — Luật shop: ưu tiên số hạt trúng Sinh/Lão MÀ không rộng quá "
        f"{sk.FS_MAX_OVER}cm; nếu không có thì BỎ phong thủy, chọn vừa tay nhất.",
        f"  => {'có' if r['fengshui_fits'] else 'KHÔNG có'} phương án vừa tay trúng Sinh/Lão",
        f"Bước 5 — Chọn {rec['count']} hạt = {rec['length_cm']}cm (dư {rec['diff_cm']}cm), "
        f"cung {rec['fengshui']}.",
    ]
    return "\n".join(lines)


def sample(prompt: str, answer: dict, think: str | None = None) -> dict:
    """Một mẫu SFT, định dạng messages (Unsloth / TRL đọc thẳng được).

    think=None  → biến thể A (đáp án thẳng)
    think=<str> → biến thể B (suy luận trong <think> rồi mới ra JSON)
    """
    content = json.dumps(answer, ensure_ascii=False)
    if think:
        content = f"<think>\n{think}\n</think>\n\n{content}"
    return {"messages": [
        {"role": "system",    "content": SYSTEM_MESSAGE},
        {"role": "user",      "content": prompt},
        {"role": "assistant", "content": content},
    ]}


# ═══════════════════════════════════════════════════════════════════
#  SPLIT — chia theo ĐƠN VỊ TRI THỨC, không chia theo mẫu
# ═══════════════════════════════════════════════════════════════════

def split_years(rng: random.Random) -> dict[str, list[int]]:
    """Mỗi NĂM chỉ nằm trong 1 split → test là năm model CHƯA TỪNG THẤY."""
    years = list(range(YEAR_MIN, YEAR_MAX + 1))
    rng.shuffle(years)
    n = len(years)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    return {
        "train": sorted(years[:n_tr]),
        "valid": sorted(years[n_tr:n_tr + n_va]),
        "test":  sorted(years[n_tr + n_va:]),
    }


def split_wrists() -> dict[str, list[float]]:
    """Train bước 0.25cm; valid/test ở giá trị XEN GIỮA → đo khả năng NỘI SUY.

    Độ phân giải 0.25 (thay vì 0.5) để phần SIZE có đủ mẫu, không bị phần MỆNH lấn át
    (201 năm vs chỉ vài chục giá trị cổ tay → mất cân bằng nặng nếu để thưa).
    """
    def rng_vals(start: float, step: float) -> list[float]:
        out, w = [], start
        while w <= 22.0001:
            out.append(round(w, 2))
            w += step
        return out

    return {
        "train": rng_vals(12.0, 0.25),   # 12.00, 12.25, 12.50, ...
        "valid": rng_vals(12.1, 1.0),    # 12.1, 13.1, ... (chưa từng thấy)
        "test":  rng_vals(12.6, 1.0),    # 12.6, 13.6, ... (chưa từng thấy)
    }


def split_canchi(rng: random.Random) -> dict[str, list[str]]:
    """Chia 60 Can Chi. Test là Can Chi model chưa từng thấy Ở DẠNG CÂU HỎI CAN CHI —
    nhưng Nạp âm của nó ĐÃ gặp qua các câu hỏi NĂM SINH trong train. Nên bài test đo
    khả năng CHUYỂN GIAO tri thức (năm→nạp âm  ⇒  can chi→nạp âm), không phải đoán mò
    một bảng tra chưa từng thấy (thứ vốn bất khả thi)."""
    all_cc = [kb._year_to_can_chi(y)["can_chi"] for y in range(1924, 1984)]
    rng.shuffle(all_cc)
    return {"train": sorted(all_cc[:42]), "valid": sorted(all_cc[42:51]),
            "test": sorted(all_cc[51:])}


CHI_LIST = kb.CHI   # 12 con giáp


def build_split(name: str, years: list[int], wrists: list[float],
                canchis: list[str], rng: random.Random, cot: bool) -> list[dict]:
    """cot=False → biến thể A (đáp án thẳng) ; cot=True → biến thể B (có <think>)."""
    rows: list[dict] = []
    is_train = name == "train"

    def mk(prompt, answer, think):
        return sample(prompt, answer, think if cot else None)

    # A. MỆNH từ NĂM SINH
    for y in years:
        for p in rng.sample(YEAR_PROMPTS, len(YEAR_PROMPTS) if is_train else 3):
            rows.append(mk(p.format(y=y), answer_year(y), think_year(y)))

    # B. MỆNH từ CAN CHI đầy đủ (xác định duy nhất)
    for cc in canchis:
        year = next(y for y in range(1924, 1984)
                    if kb._year_to_can_chi(y)["can_chi"] == cc)
        for p in rng.sample(CANCHI_PROMPTS, len(CANCHI_PROMPTS) if is_train else 2):
            rows.append(mk(p.format(cc=cc), answer_canchi(cc), think_year(year)))

    # C. CHỈ CÓ CON GIÁP → THIẾU dữ kiện → phải HỎI LẠI (không đoán).
    #    12 con giáp là hành vi, không phải tri thức cần chia tách → cho vào mọi split;
    #    valid/test dùng CÁCH HỎI khác để đo model có khái quát được hành vi không.
    for chi in CHI_LIST:
        for p in rng.sample(ZODIAC_PROMPTS, len(ZODIAC_PROMPTS) if is_train else 2):
            rows.append(mk(p.format(chi=chi), answer_zodiac_only(chi), think_zodiac(chi)))

    # D. SIZE — khách chỉ cho cổ tay, hệ thống tự chọn li
    for w in wrists:
        for p in rng.sample(WRIST_PROMPTS, len(WRIST_PROMPTS) if is_train else 2):
            rows.append(mk(p.format(w=w), answer_wrist(w), think_wrist(w)))

    # E. SIZE — khách CHỈ ĐỊNH size li (kể cả khi không khớp cổ tay)
    for w in wrists:
        for li in (6, 8, 10):
            p = rng.choice(WRIST_LI_PROMPTS)
            rows.append(mk(p.format(w=w, li=li), answer_wrist(w, li), think_wrist(w, li)))

    rng.shuffle(rows)
    return rows


def main():
    ys = split_years(random.Random(SEED))
    ws = split_wrists()
    cc = split_canchi(random.Random(SEED))
    extra_years = list(range(2101, 2161)) + list(range(1840, 1900))

    print(f"Năm     : train {len(ys['train'])} | valid {len(ys['valid'])} | test {len(ys['test'])}")
    print(f"Can Chi : train {len(cc['train'])} | valid {len(cc['valid'])} | test {len(cc['test'])}")
    print(f"Cổ tay  : train {len(ws['train'])} | valid {len(ws['valid'])} | test {len(ws['test'])}")

    # Sinh HAI BIẾN THỂ — cùng câu hỏi, cùng đáp án, chỉ khác CÓ/KHÔNG chuỗi suy luận.
    # Giữ mọi thứ khác cố định (kể cả seed) → so sánh A vs B là so sánh SẠCH, chỉ đo
    # đúng tác động của CoT.
    for cot, out_dir in ((False, Path("dataset_fengshui")),
                         (True,  Path("dataset_fengshui_cot"))):
        rng = random.Random(SEED)          # cùng seed → cùng câu hỏi, cùng thứ tự
        out_dir.mkdir(exist_ok=True)
        label = "B — CÓ suy luận <think>" if cot else "A — đáp án thẳng"
        print(f"\n{'=' * 70}\nBIẾN THỂ {label}  →  {out_dir}/")

        for name in ("train", "valid", "test"):
            rows = build_split(name, ys[name], ws[name], cc[name], rng, cot)
            path = out_dir / f"{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            avg = sum(len(r["messages"][2]["content"]) for r in rows) / len(rows)
            print(f"  {path.name:22s} → {len(rows):5d} mẫu | đáp án dài TB {avg:5.0f} ký tự")

        rows = build_split("test", extra_years, [], [], rng, cot)
        path = out_dir / "test_extrapolate.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {path.name:22s} → {len(rows):5d} mẫu | năm 1840-1899 + 2101-2160 "
              f"(NGOÀI vùng train → đo NGOẠI SUY)")

        (out_dir / "splits.json").write_text(
            json.dumps({"years": ys, "can_chi": cc, "wrists": ws,
                        "extrapolate_years": extra_years}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 70)
    print("VÍ DỤ biến thể B (có suy luận) — khách hỏi năm sinh:")
    ex = sample("Tôi sinh năm 1990, mệnh gì vậy?", answer_year(1990), think_year(1990))
    print("  Khách:", ex["messages"][1]["content"])
    print("  Model:")
    print("    " + ex["messages"][2]["content"].replace("\n", "\n    ")[:900])


if __name__ == "__main__":
    main()
