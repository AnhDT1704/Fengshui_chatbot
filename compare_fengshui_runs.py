"""
compare_fengshui_runs.py – So sánh 2 lần finetune phong thủy (biến thể A vs B).

Dùng khi train A và B ở HAI tài khoản Colab khác nhau (mỗi bên lưu vào Drive riêng):
chỉ cần tải 2 file results.json về máy rồi chạy script này — khỏi phải chia sẻ Drive.

CHẠY:
    python compare_fengshui_runs.py results_A.json results_B.json

results.json do notebook sinh ra (mục 9), chứa:
    variant, model, train_loss, test_in_range, test_extrapolate
"""

import json
import sys

MENH = ["can_chi", "napam", "element", "generating_element",
        "controlling_element", "lucky_colors", "unlucky_colors"]
SIZE = ["bead_size_li", "bead_count", "length_cm", "slack_cm", "fengshui"]


def load(path: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    print(f"  {path}  →  biến thể {d['variant']}  ({d['model']})")
    return d


def row(label: str, a, b, width=30):
    """In 1 dòng so sánh, kèm chênh lệch B−A."""
    if a is None or b is None:
        return
    d = b - a
    arrow = "↑" if d > 1 else ("↓" if d < -1 else " ")
    print(f"  {label:<{width}} {a:6.1f}%  {b:6.1f}%   {arrow}{abs(d):5.1f}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    print("Đang đọc:")
    A, B = load(sys.argv[1]), load(sys.argv[2])
    if A["variant"] == B["variant"]:
        print("\n⚠ Hai file CÙNG một biến thể — bạn tải nhầm rồi.")
        return
    if A["variant"] != "A":
        A, B = B, A                      # cho phép truyền ngược thứ tự

    for key, title in (("test_in_range",    "TRONG VÙNG (năm chưa thấy, 1900-2100) — đo NỘI SUY"),
                       ("test_extrapolate", "NGOÀI VÙNG (1840-1899 + 2101-2160) — đo NGOẠI SUY")):
        a, b = A[key], B[key]
        print("\n" + "=" * 62)
        print(f"{title}")
        print("=" * 62)
        print(f"  {'':30} {'A':>6}   {'B':>6}    B−A")
        print("  " + "-" * 52)
        print("  ── Mệnh ──")
        for f in MENH:
            row(f, a.get(f), b.get(f))
        print("  ── Size vòng ──")
        for f in SIZE:
            row(f, a.get(f), b.get(f))
        print("  ── Khác ──")
        row("JSON hợp lệ", a.get("_json_hop_le"), b.get("_json_hop_le"))
        row("BIẾT HỎI LẠI (thiếu dữ kiện)", a.get("_biet_hoi_lai"), b.get("_biet_hoi_lai"))

    # ── KẾT LUẬN ──────────────────────────────────────────────────
    # Câu hỏi nghiên cứu: model HỌC ĐƯỢC QUY LUẬT (năm-1924)%60, hay chỉ HỌC VẸT?
    # Bằng chứng: chênh lệch giữa TRONG VÙNG và NGOÀI VÙNG.
    #   chênh nhỏ  → cùng một cách trả lời cho năm quen và năm lạ → nó TÍNH, không nhớ.
    #   chênh lớn  → chỉ làm được với năm đã thấy → HỌC VẸT.
    print("\n" + "=" * 62)
    print("KẾT LUẬN — model học QUY LUẬT hay HỌC VẸT?")
    print("=" * 62)
    for tag, d in (("A (đáp án thẳng)", A), ("B (có <think>)", B)):
        i = d["test_in_range"].get("element", 0)
        o = d["test_extrapolate"].get("element", 0)
        gap = i - o
        verdict = ("HỌC ĐƯỢC QUY LUẬT" if gap < 15
                   else "chủ yếu HỌC VẸT" if gap < 40
                   else "HỌC VẸT hoàn toàn")
        print(f"  {tag:<20} trong {i:5.1f}%  |  ngoài {o:5.1f}%  |  "
              f"chênh {gap:5.1f} điểm  →  {verdict}")

    ga = A["test_in_range"].get("element", 0) - A["test_extrapolate"].get("element", 0)
    gb = B["test_in_range"].get("element", 0) - B["test_extrapolate"].get("element", 0)
    print()
    if gb < ga - 10:
        print("  → CoT (<think>) GIÚP model khái quát hoá tốt hơn hẳn: dạy nó viết ra phép")
        print("    tính thì nó thật sự TÍNH, thay vì tra trí nhớ.")
    elif ga < gb - 10:
        print("  → Bất ngờ: CoT làm KÉM ĐI. Đáng đào sâu — có thể chuỗi suy luận quá dài")
        print("    khiến model lạc, hoặc nó học thuộc luôn cả phần suy luận.")
    else:
        print("  → CoT KHÔNG tạo khác biệt rõ. Nếu cả hai đều sập ngoài vùng thì kết luận")
        print("    là: finetune KHÔNG học được quy luật — tri thức dạng LUẬT nên để CODE")
        print("    làm (như hàm _year_to_can_chi hiện tại, đúng 100% cho mọi năm).")


if __name__ == "__main__":
    main()
