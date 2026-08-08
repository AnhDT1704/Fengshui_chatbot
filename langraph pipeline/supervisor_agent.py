"""
supervisor_agent.py – Routing brain of the chatbot.

This file only owns:
  - SupervisorState
  - SUPERVISOR_SYSTEM_PROMPT
  - supervisor_node (LLM-based routing decision)
  - route_to_agent (graph conditional edge)

The full graph (wiring real sub-agents) is built in graph.py.
"""

from __future__ import annotations

import _bootstrap # noqa: F401

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
from image_turn_intent import classify_image_turn_intent
from logger import get_logger


MODEL_NAME = os.getenv("CHATBOT_MODEL", "gemini-2.5-flash")
log = get_logger("supervisor")


class SupervisorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str
    intent: str
    final_response: str
    session_id: str
    # CHAINING: supervisor lập 'plan' = danh sách agent chạy TUẦN TỰ cho 1 câu trả lời
    # (thường chỉ 1 agent; ca "ảnh + hỏi size" = [skills_agent, knowledge_base_agent]).
    # 'step' = vị trí agent kế tiếp cần chạy. Mỗi agent chạy xong tự +1 step.
    plan: list[str]
    step: int


VALID_AGENTS = {
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "order_support_agent",
    "off_platform_policy",
}

# Agent mà LLM được phép đưa vào PLAN. off_platform_policy KHÔNG nằm đây vì đã được
# bắt riêng bằng regex (lưới an toàn chính sách) trước khi gọi LLM.
PLANNABLE_AGENTS = {
    "small_talk",
    "knowledge_base_agent",
    "skills_agent",
    "order_support_agent",
}


SUPERVISOR_SYSTEM_PROMPT = """
Bạn là Supervisor của hệ thống chatbot tư vấn sản phẩm phong thủy Vạn An Group.

Nhiệm vụ: đọc tin nhắn cuối của user (kèm context hội thoại), SUY LUẬN xem câu hỏi
cần NHỮNG NĂNG LỰC nào, rồi lập KẾ HOẠCH gồm 1 hoặc NHIỀU agent phối hợp để tạo ra
câu trả lời tốt nhất cho khách.

CÁC AGENT

small_talk
  Chào hỏi, cảm ơn, tạm biệt, hỏi linh tinh không liên quan sản phẩm
  ("hello", "shop ơi", "cảm ơn nhé", "ok rồi", emoji thuần,...).
  CHỌN agent này cho mọi message ngắn mang tính giao tiếp xã giao.

knowledge_base_agent
  Tìm kiếm, lọc, so sánh, xem chi tiết SẢN PHẨM trong DB shop.
  Ví dụ: "shop có vòng aquamarine không", "vòng nào dưới 200k",
  "so sánh vòng tourmaline và mã não".
MỌI câu hỏi "shop/bên bạn CÓ BÁN / CÓ sản phẩm X không", "có loại Y không", liệt
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
  Hậu mãi / chính sách / dịch vụ khách hàng. Xử lý 2 loại:

  (A) TRẢ LỜI TRỰC TIẾP (shop có dữ liệu sẵn):
  - Bảo hành / "thay dây trọn đời"
  - Chính sách ĐỔI TRẢ / HOÀN TIỀN CHUNG ("shop có cho đổi trả không", "đổi trả mấy ngày")
  - KHUYẾN MÃI / MÃ GIẢM GIÁ / "đang sale gì"
  - THẮC MẮC sản phẩm nhận KHÔNG ĐẸP / KHÔNG SÁNG / khác màu so với ẢNH CHỤP (trấn an —
    KHÔNG phải lỗi, chỉ do ánh sáng studio).

  (B) CHUYỂN CHỦ SHOP (bot escalate → chủ shop trả lời trực tiếp; hệ thống KHÔNG có dữ liệu):
  - GIAO HÀNG / VẬN CHUYỂN / ĐƠN HÀNG: giao hoả tốc / nhanh / trong ngày, đơn vị vận chuyển,
    phí ship / COD, thời gian ship / "mấy ngày nhận được", hẹn shipper, giao sớm-gấp, "đơn em
    tới đâu rồi" / tình trạng đơn, "shop nhận đơn chưa", ĐỔI ĐỊA CHỈ giao, khiếu nại giao chậm.
  - KHIẾU NẠI SP NHẬN ĐƯỢC: lỗi / vỡ / bể / sờn / đứt, THIẾU / sai số lượng, GIAO SAI SIZE,
    GIAO NHẦM / sai mẫu / sai màu.
  - DỊCH VỤ PHỤ / YÊU CẦU ĐẶC BIỆT (shop xử lý tay): mua SỈ / số lượng lớn / giá sỉ, MIX đá-màu
    theo yêu cầu, bán HẠT LẺ / dây lẻ, ĐỔI / BỎ quà kèm, gói HỘP QUÀ / thiệp / lời chúc,
    CHARM / KHẮC TÊN / tùy chỉnh, TRÌ CHÚ / khai quang / thanh tẩy, nhờ LỰA MẪU / chụp từng mẫu,
    gộp-tách hộp, che tên khi giao, và mọi yêu cầu đặc biệt khác.
  → Với loại (B): order_support KHÔNG tự trả lời/bịa thông tin đơn-giao hàng, mà chuyển chủ shop.

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
QUY TẮC LẬP KẾ HOẠCH (PLAN)
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
- CHỈ in ra tên agent. Nhiều agent thì nối bằng " -> " theo ĐÚNG THỨ TỰ chạy.
- Tên hợp lệ: small_talk | knowledge_base_agent | skills_agent | order_support_agent
- Ví dụ hợp lệ:
    knowledge_base_agent
    skills_agent -> knowledge_base_agent
    order_support_agent
- TUYỆT ĐỐI không giải thích, không thêm ký tự nào khác ngoài tên agent (và " -> ").
"""


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
            return False # only inspect the latest human turn
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


# Note ngữ cảnh khi có ảnh — sau khi hệ thống ĐÃ LLM-classify intent (không regex).
# Không còn "có ảnh = bắt buộc KB/finetune".
_IMAGE_CONTEXT_NOTE = (
    "\n\n[NGỮ CẢNH] Tin nhắn mới nhất CÓ KÈM ẢNH. Hệ thống đã SUY LUẬN ý định text "
    "(không phải lúc nào cũng nhận diện catalog).\n"
    "- Ý định KHIẾU NẠI / TRA ĐƠN / CẦN CHỦ SHOP (ảnh = bằng chứng) → "
    "CHỈ order_support_agent (escalate). KHÔNG knowledge_base_agent.\n"
    "- Ý định TƯ VẤN / ĐỊNH DANH SP trong ảnh → knowledge_base_agent.\n"
    "- Ý định SIZE / số hạt kèm ảnh → skills_agent, hoặc "
    "skills_agent -> knowledge_base_agent nếu còn cần gắn SP."
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
    return plan[:3] # chặn an toàn: tối đa 3 agent/lượt


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
    msgs = list(state["messages"])

    # Có ảnh → LLM reasoning TEXT trước: escalate | identify | size | other.
    # Escalate: chuyển chủ shop, KHÔNG ép KB / finetune.
    image_intent = "other"
    if has_image:
        decision = classify_image_turn_intent(msgs, default="identify")
        image_intent = decision.get("intent") or "identify"
        if image_intent == "escalate":
            log.info(
                "ROUTE → order_support_agent   | img=True intent=escalate reason=%s",
                (decision.get("reason") or "")[:100],
            )
            return _plan_result(["order_support_agent"])

    # LLM tự REASONING ra KẾ HOẠCH (1 hoặc nhiều agent). Có ảnh → tiêm note intent.
    # max_tokens cao để Gemini 2.5 Flash "thinking" không ăn hết budget rồi trả rỗng.
    system_prompt = SUPERVISOR_SYSTEM_PROMPT + (_IMAGE_CONTEXT_NOTE if has_image else "")
    llm = make_llm(temperature=0, max_tokens=8192)
    response = llm.invoke(
        [SystemMessage(content=system_prompt)] + _routing_context(msgs)
    )
    raw = (response.content or "").strip()
    plan = _parse_plan(raw)

    if not plan:
        if image_intent == "size":
            plan = ["skills_agent"]
        else:
            plan = ["knowledge_base_agent"] # fallback mặc định (identify / other)

    # Ảnh + intent identify/other: cần KB nhận diện (FT/SigLIP) nếu plan chưa có.
    # Intent size: không ép KB (skills đủ); escalate đã return ở trên.
    if has_image and image_intent in ("identify", "other"):
        if "knowledge_base_agent" not in plan:
            plan.append("knowledge_base_agent")
    if has_image and image_intent == "size":
        # Ưu tiên skills; nếu LLM đã chọn KB thì giữ chuỗi skills -> KB.
        if "skills_agent" not in plan and "knowledge_base_agent" not in plan:
            plan = ["skills_agent"]
        elif "skills_agent" not in plan and "knowledge_base_agent" in plan:
            plan = ["skills_agent", "knowledge_base_agent"]

    # Bất biến: skills + KB → KB chạy CUỐI (trình bày card / chốt đáp).
    if "skills_agent" in plan and "knowledge_base_agent" in plan:
        plan = [a for a in plan if a != "knowledge_base_agent"] + ["knowledge_base_agent"]

    snippet = ""
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            c = m.content if isinstance(m.content, str) else str(m.content)
            snippet = c.replace("\n", " ")[:80]
            break
    log.info(
        "ROUTE → %-22s | img=%s intent=%s user='%s'",
        " -> ".join(plan), has_image, image_intent, snippet,
    )
    if " -> ".join(plan) != raw.lower():
        log.debug("(raw LLM output: %r)", raw)

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
