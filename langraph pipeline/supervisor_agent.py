"""
supervisor_agent.py – Routing brain of the chatbot.

This file only owns:
  - SupervisorState
  - SUPERVISOR_SYSTEM_PROMPT
  - supervisor_node  (LLM-based routing decision)
  - route_to_agent   (graph conditional edge)

The full graph (wiring real sub-agents) is built in graph.py.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import os
import re
from typing import Annotated, Sequence
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END
from langgraph.graph.message import add_messages

from gemini import make_llm
from logger import get_logger


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")
log = get_logger("supervisor")


# ═══════════════════════════════════════════════════════════════════
#  STATE
# ═══════════════════════════════════════════════════════════════════

class SupervisorState(TypedDict):
    messages:       Annotated[Sequence[BaseMessage], add_messages]
    next_agent:     str
    intent:         str
    final_response: str
    session_id:     str
    # CHAINING: supervisor lập 'plan' = danh sách agent chạy TUẦN TỰ cho 1 câu trả lời
    # (thường chỉ 1 agent; ca "ảnh + hỏi size" = [skills_agent, knowledge_base_agent]).
    # 'step' = vị trí agent kế tiếp cần chạy. Mỗi agent chạy xong tự +1 step.
    plan:           list[str]
    step:           int


VALID_AGENTS = {
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "order_support_agent",
    "other_service_agent",
    "off_platform_policy",
}

# Agent mà LLM được phép đưa vào PLAN. off_platform_policy KHÔNG nằm đây vì đã được
# bắt riêng bằng regex (lưới an toàn chính sách) trước khi gọi LLM.
PLANNABLE_AGENTS = {
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "order_support_agent",
    "other_service_agent",
}


# ═══════════════════════════════════════════════════════════════════
#  ROUTING PROMPT
# ═══════════════════════════════════════════════════════════════════

SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor của hệ thống chatbot tư vấn sản phẩm phong thủy Vạn An Group.

Nhiệm vụ: đọc tin nhắn cuối của user (kèm context hội thoại), SUY LUẬN xem câu hỏi
cần NHỮNG NĂNG LỰC nào, rồi lập KẾ HOẠCH gồm 1 hoặc NHIỀU agent phối hợp để tạo ra
câu trả lời tốt nhất cho khách.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CÁC AGENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

small_talk
  Chào hỏi, cảm ơn, tạm biệt, hỏi linh tinh không liên quan sản phẩm
  ("hello", "shop ơi", "cảm ơn nhé", "ok rồi", emoji thuần,...).
  CHỌN agent này cho mọi message ngắn mang tính giao tiếp xã giao.

knowledge_base_agent
  Tìm kiếm, lọc, so sánh, xem chi tiết SẢN PHẨM trong DB shop.
  Ví dụ: "shop có vòng aquamarine không", "vòng nào dưới 200k",
  "so sánh vòng tourmaline và mã não".
  ⚠️ MỌI câu hỏi "shop/bên bạn CÓ BÁN / CÓ sản phẩm X không", "có loại Y không", liệt
  kê sản phẩm loại nào đó → LUÔN knowledge_base_agent để TRA DB THẬT. TUYỆT ĐỐI KHÔNG
  tự suy đoán "shop không bán X" rồi đẩy đi nơi khác — kể cả khi X nghe lạ / không
  giống đồ phong thủy (vd "dầu gió", "tinh dầu", "nhang", "than xông"): shop có thể có
  trong DB, phải để KB tra rồi mới biết.
  CŨNG xử lý HƯỚNG DẪN SỬ DỤNG & BẢO QUẢN sản phẩm: vòng rộng/chật chỉnh sao,
  đứt dây/thay dây, bảo quản trầm hương, đeo có đụng nước được không, cách đeo.
  CŨNG xử lý TƯ VẤN THEO MỆNH / TUỔI / NĂM SINH (Can Chi Nạp Âm → mệnh → màu/đá
  hợp → lọc sản phẩm). Ví dụ: "mình sinh 1990 hợp đá nào", "mệnh Hỏa đeo màu gì",
  "tuổi Tý nên đeo vòng gì".
  CŨNG xử lý MỌI MESSAGE CÓ KÈM ẢNH (tìm SP giống ảnh, hỏi phong thủy về SP trong
  ảnh) HOẶC yêu cầu XEM ẢNH sản phẩm của shop.
  CŨNG xử lý SỐ HẠT VÒNG MẶC ĐỊNH theo size & ý nghĩa phong thủy CHUNG (KHÔNG kèm
  số đo cổ tay): "X li (mm) bao nhiêu hạt", "vòng này bao nhiêu hạt", "số hạt theo
  Sinh-Lão-Bệnh-Tử nghĩa là gì". (KHÁC: hễ khách ĐƯA SỐ ĐO CỔ TAY (Xcm) để tính
  size / số hạt cho vừa — kể cả khi muốn 1 size li cụ thể — thì → skills_agent.)

skills_agent
  Câu hỏi cần TÍNH TOÁN hoặc TƯ VẤN CHUYÊN MÔN:
  - Tính size vòng tay từ cm cổ tay (cổ tay Xcm → đeo size mấy li)
  - Tính SỐ HẠT cho vừa cổ tay (cổ tay Xcm + size li → xâu mấy hạt), kể cả khi khách
    muốn 1 size li KHÔNG khớp cổ tay; cân nhắc Sinh-Lão-Bệnh-Tử và thêm/bớt hạt cho vừa
  - ƯỚC LƯỢNG size khi khách KHÔNG đo cổ tay mà chỉ cho CHIỀU CAO / CÂN NẶNG / giới
    tính / "tay to-nhỏ" (vd "nữ 1m55 50kg đeo size mấy", "cao 1m7 nặng 60 thì mấy li")
  - Tư vấn quà tặng theo người nhận / dịp
  LƯU Ý: "shop có bán X không" KHÔNG thuộc skills — luôn để knowledge_base_agent tra DB
  trước (web_search chỉ là công cụ KB/skills tự dùng SAU khi đã chắc DB không có).

order_support_agent
  CHỈ các câu CHÍNH SÁCH CHUNG shop có sẵn dữ liệu:
  - Bảo hành / "thay dây trọn đời"
  - Chính sách ĐỔI TRẢ / HOÀN TIỀN CHUNG (vd "shop có cho đổi trả không", "đổi trả mấy ngày")
  - KHUYẾN MÃI / MÃ GIẢM GIÁ / "đang sale gì"
  - THẮC MẮC sản phẩm nhận KHÔNG ĐẸP / KHÔNG SÁNG / khác màu so với ẢNH CHỤP (trấn an —
    đây KHÔNG phải lỗi, chỉ do ánh sáng studio).
  (KHÔNG xử lý giao hàng/vận chuyển/tra đơn/khiếu nại lỗi-thiếu-sai — xem other_service_agent.)

other_service_agent
  Mọi yêu cầu shop phải xử lý THỦ CÔNG / hệ thống KHÔNG có dữ liệu → chuyển thẳng cho
  CHỦ SHOP trả lời trực tiếp (bot không tự trả lời). GỒM:

  (1) GIAO HÀNG / VẬN CHUYỂN / ĐƠN HÀNG (hệ thống KHÔNG theo dõi được đơn — chỉ chủ shop
      tra trên Shopee): giao hoả tốc / giao nhanh / giao trong ngày / giao buổi sáng,
      phương thức giao / đơn vị vận chuyển / phí ship / COD, thời gian ship / "mấy ngày
      nhận được", hẹn / liên lạc shipper, yêu cầu giao sớm-giao gấp, "đơn em tới đâu rồi"
      / tình trạng đơn, "shop nhận đơn chưa", ĐỔI ĐỊA CHỈ giao, khiếu nại giao chậm.

  (2) KHIẾU NẠI SẢN PHẨM NHẬN ĐƯỢC: lỗi / vỡ / bể / sờn / đứt, THIẾU hàng / sai số lượng
      so với quảng cáo, GIAO SAI SIZE / sai kích thước, GIAO NHẦM / sai mẫu / sai màu.

  (3) DỊCH VỤ PHỤ / YÊU CẦU ĐẶC BIỆT (shop xử lý tay):
  - Mua SỈ / số lượng lớn / nhập hàng để bán lại / xin giá sỉ / combo số lượng
  - MIX đá / mix màu / xếp thứ tự hạt THEO YÊU CẦU riêng của khách
  - Bán HẠT LẺ / dây lẻ / phụ kiện lẻ để khách tự xâu
  - ĐỔI / BỎ quà tặng kèm (vd đổi quà thành dây xâu, không lấy quà)
  - Đóng HỘP QUÀ / gói quà / kèm thiệp / viết lời chúc / túi đựng đặc biệt
  - CHARM / móc khóa / KHẮC TÊN / chạm khắc / tùy chỉnh sản phẩm theo ý khách
  - TRÌ CHÚ / khai quang / thanh tẩy / làm phép theo yêu cầu
  - Nhờ LỰA MẪU giúp / chụp ảnh từng mẫu cho khách chọn
  - Gộp / tách hộp khi giao nhiều đơn, che tên khi giao, và mọi yêu cầu đặc biệt khác
  NGUYÊN TẮC: nếu là yêu cầu KHÔNG nằm trong năng lực dữ liệu của các agent khác (sản
  phẩm, mệnh/tuổi, size theo cổ tay, chính sách bảo hành/đổi trả/khuyến mãi) → đây là
  việc của other_service_agent (chuyển chủ shop). ĐẶC BIỆT mọi thứ về GIAO HÀNG / ĐƠN
  HÀNG / KHIẾU NẠI SẢN PHẨM LỖI-THIẾU-SAI đều → other_service_agent.

off_platform_policy
  Khách xin THÔNG TIN LIÊN HỆ / ĐỊA CHỈ của shop, hoặc rủ GIAO DỊCH NGOÀI Shopee.
  Theo quy định Shopee, shop KHÔNG cung cấp và phải từ chối khéo. Gồm:
  - Xin ĐỊA CHỈ shop / hỏi "shop ở đâu" / ghé cửa hàng xem-mua-lấy-đo tay trực tiếp:
    "shop ở đâu Đà Nẵng", "cho mình xin địa chỉ qua mua trực tiếp", "em ghé cửa hàng
    đo tay được không", "shop mình ở đâu mình chạy qua lấy".
  - Xin SỐ ĐIỆN THOẠI / SĐT / ZALO / FACEBOOK / kết bạn:
    "cho mình xin sđt có zalo", "shop có facebook không", "bạn cho mình số đt với".
  - Rủ mua / giao dịch NGOÀI sàn Shopee (né phí sàn/ship):
    "đi đơn ngoài sàn được không", "shop không bán ở ngoài à", "gửi địa chỉ rồi nhận
    hàng chuyển khoản được không", "book ship ngoài giúp".
  CHỌN agent này cho MỌI câu xin liên hệ/địa chỉ shop hoặc rủ giao dịch ngoài Shopee.
  LƯU Ý: khách CHO địa chỉ NHẬN HÀNG của họ (để ship qua Shopee) thì KHÔNG phải case
  này → đó là order_support_agent.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC LẬP KẾ HOẠCH (PLAN)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Phân tích câu hỏi cần những NĂNG LỰC gì → chọn 1 HOẶC NHIỀU agent.
2. Chào / cảm ơn / emoji thuần → small_talk (1 agent).
3. PHỐI HỢP NHIỀU AGENT (chuỗi) khi 1 câu cần >1 năng lực. Các agent chạy TUẦN TỰ:
   agent ĐỨNG SAU đọc được kết quả của agent trước và là người SOẠN câu trả lời cuối
   cho khách. → Đặt agent "tính toán / hỗ trợ" TRƯỚC, agent "trình bày / chốt đáp" SAU.
   Ví dụ điển hình:
   - Ảnh sản phẩm + hỏi size theo cổ tay / chiều cao / cân nặng:
       skills_agent -> knowledge_base_agent
       (skills tính số hạt CHÍNH XÁC trước; KB nhận diện SP qua ảnh + trình bày card
        và DÙNG số hạt skills đã tính)
   - Khách mô tả/đưa số đo cổ tay để tính size cho 1 sản phẩm CỤ THỂ (đã biết SP):
       skills_agent -> knowledge_base_agent (nếu cần trình bày lại card SP)
   - Chỉ định danh / hỏi về sản phẩm qua ẢNH (không hỏi size): knowledge_base_agent
   - Chỉ tính size cổ tay / tư vấn quà / web-search (không cần định danh SP): skills_agent
4. Nếu chỉ cần 1 năng lực → trả về đúng 1 agent (đa số trường hợp).
5. Khi phân vân KB vs skills:
   - Mô tả/lọc sản phẩm, HỎI SHOP CÓ BÁN / CÓ sản phẩm gì không (bất kể tên lạ hay
     quen), tư vấn theo mệnh/tuổi/năm sinh, có ảnh → knowledge_base_agent
   - Cần TÍNH số hạt/size theo cổ tay-vóc dáng, hoặc tư vấn quà tặng → skills_agent
   - KHÔNG tự phán đoán "shop không bán X" để né KB — luôn để KB tra DB trước.
   - Nếu cần CẢ HAI (vd ảnh + hỏi size) → chuỗi như mục 3.

ĐỊNH DẠNG TRẢ VỀ
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- CHỈ in ra tên agent. Nhiều agent thì nối bằng " -> " theo ĐÚNG THỨ TỰ chạy.
- Tên hợp lệ: small_talk | knowledge_base_agent | skills_agent | order_support_agent
  | other_service_agent
- other_service_agent dùng MỘT MÌNH (không ghép chuỗi với agent khác).
- Ví dụ hợp lệ:
    knowledge_base_agent
    skills_agent -> knowledge_base_agent
    order_support_agent
    other_service_agent
- TUYỆT ĐỐI không giải thích, không thêm ký tự nào khác ngoài tên agent (và " -> ").
"""


# ═══════════════════════════════════════════════════════════════════
#  SUPERVISOR NODE
# ═══════════════════════════════════════════════════════════════════

def _routing_context(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Strip ToolMessages so routing LLM only sees user / assistant turns."""
    return [m for m in messages if not isinstance(m, ToolMessage)]


def _latest_human_has_image(messages: Sequence[BaseMessage]) -> bool:
    """True if the most recent HumanMessage carries an image part.

    Gemini multimodal messages store images as content parts of shape
    {"type": "image_url", ...}. Detecting this lets us route images
    deterministically to knowledge_base_agent without an LLM round-trip.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, list):
                return any(
                    isinstance(part, dict) and part.get("type") == "image_url"
                    for part in c
                )
            return False  # only inspect the latest human turn
    return False


# Khách xin liên hệ/địa chỉ shop hoặc rủ giao dịch ngoài Shopee → trả lời cố định.
# Detect bằng regex (high-precision) để bắt chắc + khỏi tốn 1 lượt LLM. Các ca khó
# (keyword bỏ sót) vẫn được supervisor LLM route nhờ mục off_platform_policy trong prompt.
_OFF_PLATFORM_RE = [
    re.compile(p, re.IGNORECASE) for p in [
        # Xin địa chỉ / ghé mua-xem-lấy-đo trực tiếp tại shop
        r"(shop|cửa\s*hàng|cừa\s*hàng|tiệm)\s*\w*\s*(ở|tại|chỗ|bán\s*ở)\s*đâu",
        r"\bbán\s*ở\s*đâu",
        r"địa\s*chỉ\s*(của\s*)?(shop|cửa\s*hàng|tiệm|bên|mua)",
        r"(xin|cho)\s*\w*\s*địa\s*chỉ",
        r"(qua|ghé|ghe|đến|den|chạy\s*qua|tới)\s*(\w+\s*){0,3}(xem|mua|lấy|lay|trực\s*tiếp|cửa\s*hàng)",
        r"(xem|mua|lấy|nhận|đo\s*tay)\s*(\w+\s*){0,2}trực\s*tiếp",
        r"ghé\s*(\w+\s*){0,2}(cửa\s*hàng|shop|tiệm)",
        # Thông tin liên hệ
        r"số\s*(điện\s*thoại|đt|dt)",
        r"\bsđt\b", r"\bsdt\b",
        r"\bzalo\b",
        r"\bfacebook\b", r"\bfb\b",
        r"kết\s*bạn",
        # Giao dịch ngoài Shopee
        r"(đơn|order|mua|bán|giao\s*dịch|ship|đặt|gửi)\s*(\w+\s*){0,3}(ngoài|bên\s*ngoài)",
        r"ngoài\s*(sàn|shopee|shoppe)",
        r"(không|ko|k)\s*bán\s*(ở\s*)?ngoài",
        r"chuyển\s*khoản",
    ]
]


def _latest_human_text(messages: Sequence[BaseMessage]) -> str:
    """Văn bản của HumanMessage mới nhất (gom phần text nếu là message đa phương thức)."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            c = m.content
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return " ".join(
                    p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            return ""
    return ""


def _is_off_platform_request(messages: Sequence[BaseMessage]) -> bool:
    text = _latest_human_text(messages).strip().lower()
    if not text:
        return False
    return any(rx.search(text) for rx in _OFF_PLATFORM_RE)


# Note ngữ cảnh tiêm vào prompt routing khi tin nhắn CÓ ẢNH — để LLM tự reasoning
# (không hard-code plan). Chỉ là SỰ THẬT khách quan + gợi ý, LLM vẫn tự quyết plan.
_IMAGE_CONTEXT_NOTE = (
    "\n\n[NGỮ CẢNH] Tin nhắn mới nhất của khách CÓ KÈM ẢNH sản phẩm. Việc nhận diện / "
    "xem ảnh cần knowledge_base_agent. Nếu khách CÒN hỏi tính size (cổ tay / chiều cao / "
    "cân nặng / số hạt cho vừa) thì hãy PHỐI HỢP: skills_agent -> knowledge_base_agent."
)


def _parse_plan(raw: str) -> list[str]:
    """Trích danh sách agent (theo thứ tự xuất hiện) từ output LLM.

    LLM được yêu cầu in 'a -> b'. Ta quét vị trí xuất hiện đầu tiên của từng tên
    agent hợp lệ rồi sắp theo vị trí → giữ đúng thứ tự chuỗi. Bỏ trùng, chặn độ dài.
    """
    raw = (raw or "").lower()
    found = [(raw.find(name), name) for name in PLANNABLE_AGENTS if raw.find(name) != -1]
    found.sort()
    plan = [name for _, name in found]
    return plan[:3]  # chặn an toàn: tối đa 3 agent/lượt


def _plan_result(plan: list[str]) -> dict:
    """Đóng gói kết quả định tuyến: 'plan' chạy tuần tự, bắt đầu từ step 0."""
    return {"plan": plan, "step": 0, "next_agent": plan[0], "intent": "+".join(plan)}


def supervisor_node(state: SupervisorState) -> SupervisorState:
    # Xin liên hệ/địa chỉ shop hoặc rủ giao dịch ngoài Shopee → off_platform_policy
    # (node trả câu cố định). Lưới an toàn chính sách, ưu tiên cao nhất.
    if _is_off_platform_request(state["messages"]):
        log.info("ROUTE → %-22s | (off-platform/contact request)", "off_platform_policy")
        return _plan_result(["off_platform_policy"])

    has_image = _latest_human_has_image(state["messages"])

    # LLM tự REASONING ra KẾ HOẠCH (1 hoặc nhiều agent, có thứ tự). Nếu có ảnh, tiêm
    # 'sự thật có ảnh' vào prompt để LLM cân nhắc — KHÔNG hard-code plan.
    # max_tokens cao để Gemini 2.5 Flash "thinking" không ăn hết budget rồi trả rỗng.
    system_prompt = SUPERVISOR_SYSTEM_PROMPT + (_IMAGE_CONTEXT_NOTE if has_image else "")
    llm = make_llm(temperature=0, max_tokens=8192)
    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + _routing_context(state["messages"])
    )
    raw  = (response.content or "").strip()
    plan = _parse_plan(raw)

    # ── Lưới an toàn ──────────────────────────────────────────────
    if not plan:
        plan = ["knowledge_base_agent"]                       # fallback mặc định
    if "other_service_agent" in plan:
        plan = ["other_service_agent"]                        # dịch vụ phụ → chạy MỘT MÌNH
    else:
        if has_image and "knowledge_base_agent" not in plan:
            plan.append("knowledge_base_agent")               # ảnh phải có KB xử lý
        # Bất biến cấu trúc: nếu chuỗi có cả skills + KB thì KB phải chạy CUỐI (KB là
        # agent trình bày card sản phẩm tốt nhất → để nó soạn câu trả lời cuối).
        if "skills_agent" in plan and "knowledge_base_agent" in plan:
            plan = [a for a in plan if a != "knowledge_base_agent"] + ["knowledge_base_agent"]

    snippet = ""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            snippet = c.replace("\n", " ")[:80]
            break
    log.info("ROUTE → %-22s | img=%s user='%s'", " -> ".join(plan), has_image, snippet)
    if " -> ".join(plan) != raw.lower():
        log.debug("       (raw LLM output: %r)", raw)

    # Only return changed keys — do NOT spread state (tránh add_messages nhân đôi).
    return _plan_result(plan)


def route_to_agent(state: SupervisorState) -> str:
    """Conditional edge dùng CHUNG cho supervisor và mọi agent node.

    Trả về tên agent kế tiếp trong 'plan' (theo 'step'), hoặc END khi đã chạy hết.
    Mỗi agent node tự tăng 'step' sau khi chạy, nên sau agent cuối → END.
    """
    plan = state.get("plan") or []
    step = state.get("step", 0)
    if 0 <= step < len(plan):
        return plan[step]
    return END
